import dataclasses
import pprint
import time
from functools import partial
import json
from multiprocessing import Pool

import h5py
import mlxu
from ml_collections.config_dict import config_dict
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer
from typing import Any
import numpy as np 
import jax
from transformer import tasks, text_dataset
import seqio
import tensorflow as tf
from jax.sharding import PartitionSpec as P
from EasyLM.jax_utils import get_jax_mesh
from EasyLM.models.rpt.rpt_model import RetrieverSupervision
import gin



def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id

def split_batch_dimension(inputs: Any, num_replicas: int) -> Any:
  """Splits the leading batch dimension.

  Given inputs of shape [num_replicas * batch_size, ...], it will reshape
  them to [num_replicas, batch_size, ...].  This operation is intended to be
  used right before calling pmap, which will eliminate the num_replicas
  dimension.

  Args:
    inputs: Tuple of inputs to split.
    num_replicas: Number of replicas.

  Returns:
    inputs with extra batch dimension.
  """

  def split_batch_dim(x):
    assert x.ndim > 0
    if (x.shape[0] % num_replicas) != 0:
      raise ValueError(f"Can't split {x.shape} into {num_replicas} replicas.")
    batch_size = x.shape[0] // num_replicas
    split_shape = [num_replicas, batch_size] + list(x.shape[1:])
    return np.reshape(x, split_shape)

  return jax.tree_map(split_batch_dim, inputs)
import einops

# def split_for_pjit(inputs: Any, n_local_devices: int, n_procs:int) -> Any:

#   def split_batch_dim(x):
#     assert x.ndim > 0
#     if (x.shape[0] % n_local_devices) != 0:
#       raise ValueError(f"Can't split {x.shape} into {num_replicas} replicas.")
#     batch_size = x.shape[0] // n_local_devices
    
    
#     return einops.rearrange(x,"b (n d) ... -> (n b) d ...",n=n_local_devices,b=n_procs)

#   return jax.tree_map(split_batch_dim, inputs)


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'seqio'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.seqio_dataset = SeqioDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'huggingface':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'seqio':
            print("using seqio dataset and ignoring tokenizer and text_processor")
            return SeqioDataset(config.seqio_dataset, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class SeqioDataset(object):
    """ Seqio dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.name = 'codeparrot'
        config.split = 'train'
        config.seq_length = 2048
        config.shuffle_buffer_size = 10000
        config.document_length = None
        config.chunk_size = None
        config.num_scored_neighbors = None
        config.is_epr = False
        config.offset = 0
        

        
        
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, model_config=None):
        self.config = self.get_default_config(config)
        del config
        if model_config is not None:
            self.config.num_scored_neighbors = model_config.num_scored_neighbors
            self.config.chunk_size = model_config.chunk_size
            self.config.document_length = model_config.document_length
        
            
        self.config.num_document_chunks = self.config.document_length//self.config.chunk_size
        print("Shards: %d of %d", jax.process_index(), jax.process_count())
        shard_info = seqio.ShardInfo(index=jax.process_index(),
                                  num_shards=jax.process_count())
        dataset_name = f"{self.config.name}neox_retro_nn20_f20_entirebook_qa_seq1024_16384"
        if self.config.is_epr:
            dataset_name = f"{dataset_name}_wloss"
            
        self._task = seqio.get_mixture_or_task(dataset_name)
        kwargs = dict()
        if "all_v1" not in dataset_name:
            kwargs['shuffle_buffer_size']= self.config.shuffle_buffer_size
        dataset = self._task.get_dataset(split=self.config.split,
                                         shuffle= self.config.split == "train",
                                         sequence_length=None,
                                         shard_info=shard_info,
                                         num_epochs=None if self.config.split == "train" else 1,
                                         seed=42,
                                         **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.bos_token_id = self._tokenizer.bos_token_id
        


        feature_lengths = {
            'targets': self.config.document_length,
            'loss_masks': self.config.document_length,
            'attention_mask': self.config.document_length,
            'nei_idx': self.config.num_scored_neighbors*self.config.num_document_chunks,
            'nei_scores': self.config.num_scored_neighbors*self.config.num_document_chunks,
        }
        

        def create_mask(x):
            return dict(loss_masks=tf.ones_like(x['targets']), **x)
        dataset = dataset.filter(lambda x:tf.shape(x["targets"])[0]>=int(self.config.document_length*0.9))
        dataset = dataset.map(create_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = seqio.preprocessors.append_eos_after_trim(dataset,
                                                    self._task.output_features,
                                                    sequence_length={ **feature_lengths, "book_id": None,})

        dataset = seqio.utils.trim_and_pad_dataset(dataset,feature_lengths)
        def format_example(x):
            targets = x.pop('targets')
            x['target_tokens'] = targets
            x['input_tokens'] = _shift_right_by_one(targets,bos_id=self.bos_token_id)
            x['attention_mask'] = _shift_right_by_one(x["loss_masks"],bos_id=1)
            return x
        dataset  = dataset.map(format_example,num_parallel_calls=tf.data.AUTOTUNE)
        self._dataset  = dataset.prefetch(tf.data.AUTOTUNE)
        
    def format_ret_sup(self,example):
        example["retriever_supervision"] = RetrieverSupervision(example.pop('nei_scores'),example.pop('nei_idx'))
        return example
        
    def __iter__(self):
        total_tokens = 0
        while True:

            for index, example in enumerate(self._dataset.as_numpy_iterator()):
                book_id = example.pop('book_id')
                example = split_batch_dimension(example, jax.local_device_count())
                example = self.format_ret_sup(example)
                input_tokens = example['input_tokens'].copy()
                input_tokens[:, 0] = self.bos_token_id
                example['input_tokens'] = input_tokens
                
                
                total_tokens += int(example['loss_masks'].sum())
                metrics = {
                    'dataset_example_index': index,
                    'dataset_total_tokens': total_tokens,
                }
                yield example, metrics

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return None

    @property
    def text_processor(self):
        return None

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
