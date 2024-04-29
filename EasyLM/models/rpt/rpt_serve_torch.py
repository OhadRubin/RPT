import os
import transformers
from EasyLM.jax_utils import unfreeze

if "DEBUG" in os.environ:
    import debugpy

    debugpy.listen(5678)
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

import mlxu
import optax
from transformers import GenerationConfig
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.models.rpt.rpt_model_torch import RPTConfig, RPTForCausalLM, RPTLowcoderRetrieverEncodedOutput, \
    EncodedNeighbors
from EasyLM.models.rpt.memory_torch import Memory
import gin
import tqdm
import absl
import torch

absl.flags.DEFINE_multi_string(
    'gin_file', None, 'List of paths to the config files.')
absl.flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='bf16',
    input_length=1024,
    seq_length=2048,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    add_bos_token=True,
    load_rpt_config='',
    load_checkpoint='',
    # tokenizer=RPTConfig.get_tokenizer_config(),
    lm_server=LMServer.get_default_config(),
    add_outputs=False,
    single_model=True,
    nearest_chunk_distance=16,
    override_list="",
    max_new_tokens=2048,
    iterative_mode=True,
    dense_mem=True,
    num_neighbors=2,
)

import numpy as np


def prepare_prefix(prefix_tokenizer, text, input_length, add_bos_token, device):
    inputs = prefix_tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=input_length,
        return_tensors='np',
    )
    input_tokens = inputs.input_ids
    input_mask = inputs.attention_mask
    if add_bos_token:
        input_tokens[:, 0] = prefix_tokenizer.bos_token_id
        input_mask[:, 0] = 1
    batch = dict(
        input_tokens=torch.Tensor(input_tokens).type(torch.int).to(device),
        input_mask=torch.Tensor(input_mask).type(torch.int).to(device),
    )
    return batch


def apply_forward_upcoder(hf_model,
                          input_tokens,
                          input_mask,
                          output_tokens,
                          output_mask,
                          upcoder_input
                          ):
    outputs, past_key_values = hf_model(input_tokens, attention_mask=input_mask,
        upcoder_input=upcoder_input,
        deterministic=True,
        mutable=['cache', 'intermediates']
    )
    output = process_logits(output_tokens, output_mask, outputs.logits)
    #past_key_values = unfreeze(past_key_values).get("cache", None)
    return output, past_key_values


def apply_forward_loglikelihood(hf_model,
                                input_tokens,
                                input_mask,
                                output_tokens,
                                output_mask,
                                ):
    outputs, past_key_values = hf_model(
        input_tokens, attention_mask=input_mask, deterministic=True
    )
    output = process_logits(output_tokens, output_mask, outputs.logits)
    #past_key_values = unfreeze(past_key_values).get("cache", None)

    return output, past_key_values


def apply_forward_lowcoder(hf_model, input_tokens, input_mask, **kwargs):
    # TODO: Apply and other parameters
    # TODO: Output attention
    outputs, past_key_values = hf_model._lowcoder_forward(
        input_ids=input_tokens,
        attention_mask=input_mask,
        deterministic=True,
        **kwargs,
    )

    return outputs, past_key_values


def apply_forward_augment(hf_model, hidden_states, neighbor_hidden_states, neighbor_mask, past_key_values):
    outputs, past_key_values = hf_model._augment_forward(
        hidden_states=hidden_states,
        neighbor_hidden_states=neighbor_hidden_states,
        neighbor_mask=neighbor_mask,
        deterministic=True,
        layer_past=past_key_values['augment'],
        init_cache=True, # TODO: Really??
    )
    #past_key_values = unfreeze(past_key_values).get("cache", None)
    return outputs, {'augment': past_key_values}


def _loglikelihood_rolling(tokenizer, hf_model, text, func, nearest_chunk_distance, num_neighbors=2, input_length=1024,
                           verbose=True, return_scores=False):
    memory = Memory(chunk_size=64, num_neighbors=num_neighbors, nearest_chunk_distance=nearest_chunk_distance,
                    return_scores=return_scores)
    #params.update(cache=jax.tree_map(lambda x: jnp.zeros_like(x), params['cache']))
    # TODO: cache intitialization

    loglikelihood_list = []
    total_loglikelihood = 0.0
    total_is_greedy = True
    metadata_list = tuple()
    token_count = np.zeros((len(text),), dtype=np.int32)

    for batch in rolling_iterator(tokenizer, text, input_length):
        token_count += batch['output_mask'].sum(-1)

        (loglikelihood, is_greedy), metadata = func(
            hf_model, batch, memory
        )
        metadata_list += (metadata,)
        total_loglikelihood += loglikelihood
        loglikelihood_list.append(loglikelihood.item())
        total_is_greedy = np.logical_and(is_greedy, total_is_greedy)
    if verbose:
        print(loglikelihood_list)
    return total_loglikelihood, total_is_greedy, token_count, metadata_list


def create_forward_loglikelihood(config, low_fwd, up_fwd, fwd):
    def forward_loglikelihood_no_mem(params, batch, memory):
        outputs, past_key_values = fwd(params, batch)
        return outputs, past_key_values

    def forward_loglikelihood_w_mem(params, batch, memory):
        outputs, past_key_values = low_fwd(params, batch)
        params.update(cache=past_key_values)

        neighbor_hidden_states, neighbor_mask, metadata, *_ = memory.add(
            input_tokens=batch["input_tokens"],
            encoded_hidden_states=outputs.encoded_hidden_states,
            key_chunks=outputs.key_chunks,
            query_chunks=outputs.query_chunks,
        )
        batch.update(
            upcoder_input=RPTLowcoderRetrieverEncodedOutput(
                hidden_states=outputs.original_hidden_states,
                attention_mask=outputs.attention_mask,
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask
            )
        )
        outputs, past_key_values = up_fwd(params, batch)
        params.update(cache=past_key_values)
        return outputs, metadata

    if config.cca_freq == 0:
        return forward_loglikelihood_no_mem
    else:
        return forward_loglikelihood_w_mem


def filter_(intermediates):
    def cont(k):
        for s in ["layers/0", "retriever", "/neighbor_"]:
            if s in k:
                return True
        return False

    return {k: v for k, v in intermediates.items() if cont(k)}


def postproc_output(tokenizer, output, output_text, verbose=False):
    new_output_text = []
    for old_text, text in zip(output_text, list(tokenizer.batch_decode(output))):
        if tokenizer.eos_token in text:
            text = text.split(tokenizer.eos_token, maxsplit=1)[0]
        if verbose:
            print(text, end='')
        new_output_text.append(old_text + (text,))

    return new_output_text


def process_logits(output_tokens, output_mask, logits):
    loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
        logits, output_tokens
    )
    loglikelihood = torch.sum(loglikelihood * output_mask, dim=-1)
    match_count = torch.sum(
        (torch.argmax(logits, dim=-1) == output_tokens) * output_mask,
        dim=-1
    )
    total = torch.sum(output_mask, dim=-1)
    is_greedy = match_count == total
    return loglikelihood, is_greedy


def rolling_iterator(tokenizer, text, input_length):
    inputs = tokenizer(
        text,
        padding='longest',
        truncation=False,
        max_length=np.iinfo(np.int32).max,
        return_tensors='np',
    )
    batch_size = inputs.input_ids.shape[0]

    output_tokens = inputs.input_ids
    attention_mask = inputs.attention_mask

    if output_tokens.shape[1] < input_length:
        padding_length = input_length - output_tokens.shape[1]
        pad_tokens = np.full(
            (batch_size, padding_length), tokenizer.pad_token_id, dtype=np.int32
        )
        output_tokens = np.concatenate([output_tokens, pad_tokens], axis=-1)
        pad_mask = np.zeros(
            (batch_size, padding_length), dtype=inputs.attention_mask.dtype
        )
        attention_mask = np.concatenate([attention_mask, pad_mask], axis=-1)

    bos_tokens = np.full(
        (batch_size, 1), tokenizer.bos_token_id, dtype=np.int32
    )
    input_tokens = np.concatenate([bos_tokens, output_tokens[:, :-1]], axis=-1)
    # bos_mask = np.ones((batch_size, 1), dtype=inputs.attention_mask.dtype)
    total_input_length = output_tokens.shape[1]

    # Sliding window
    for i in tqdm.tqdm(range(0, total_input_length, input_length)):
        # Last window
        # TODO: there is a bug here, for ABC, the last window should be BC, not C0 not BC with B padded.
        if i + input_length > total_input_length:
            last_output_mask = np.copy(attention_mask[:, -input_length:])
            last_output_mask[:, :i - total_input_length] = 0.0

            batch = dict(
                input_tokens=input_tokens[:, -input_length:].astype(int),
                output_tokens=output_tokens[:, -input_length:].astype(int),
                input_mask=attention_mask[:, -input_length:].astype(int),
                output_mask=last_output_mask.astype(int),
            )

        # Normal window
        else:
            batch = dict(
                input_tokens=input_tokens[:, i:i + input_length].astype(int),
                output_tokens=output_tokens[:, i:i + input_length].astype(int),
                input_mask=attention_mask[:, i:i + input_length].astype(int),
                output_mask=attention_mask[:, i:i + input_length].astype(int),
            )
        yield batch



def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    #jax.distributed.initialize()

    prefix_tokenizer = RPTConfig.get_tokenizer(truncation_side='left', padding_side='left')
    tokenizer = RPTConfig.get_tokenizer(truncation_side='right', padding_side='right')

    rpt_config = RPTConfig.load_config(FLAGS.load_rpt_config)
    config = RPTConfig()
    override_dict = {key: getattr(config, key) for key in FLAGS.override_list.split(",")}
    rpt_config.update(override_dict)
    state, params = StreamingCheckpointer.load_trainstate_checkpoint(
        FLAGS.load_checkpoint, disallow_trainstate=True
    )

    print(rpt_config)
    params = unfreeze(params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hf_model = RPTForCausalLM(rpt_config, device=device)
    hf_model.to(device)


    # TODO: Cringe
    gin.clear_config()

    from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model

    for key_to_modify in ['lowcoder', 'upcoder']:
        layers = params['params']['transformer'][key_to_modify]['layers']
        del params['params']['transformer'][key_to_modify]['layers']
        params['params']['transformer'][key_to_modify]['layers'] = {'blocks': layers }

    load_flax_weights_in_pytorch_model(hf_model, params['params'])

    hf_model.save_pretrained('rpt-torch-1')
    hf_model.from_pretrained('rpt-torch-1')

    _forward_upcoder = apply_forward_upcoder
    _forward_loglikelihood = apply_forward_loglikelihood
    _forward_lowcoder = apply_forward_lowcoder
    _forward_augment = apply_forward_augment

    forward_loglikelihood = create_forward_loglikelihood(rpt_config, _forward_lowcoder, _forward_upcoder,
                                                         _forward_loglikelihood)


    def _forward_generate(batch, max_new_tokens, temperature, sample=True, past_key_values=None):
        if sample:
            generate_kwargs = dict(
                logits_processor=transformers.LogitsProcessorList(
                    [transformers.TemperatureLogitsWarper(temperature)]
                ),
                generation_config=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=FLAGS.do_sample,
                    num_beams=FLAGS.num_beams,
                    top_k=FLAGS.top_k,
                    top_p=FLAGS.top_p,
                    return_dict_in_generate=True,
                ))
        else:
            generate_kwargs = dict(generation_config=GenerationConfig(
                max_new_tokens=FLAGS.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
            ),
            )

        output, encoded_lowcoder_states = hf_model.generate(
            torch.Tensor(batch['input_tokens']),
            attention_mask=torch.Tensor(batch['input_mask']).type(torch.int),
            encoded_neighbors=batch.get("encoded_neighbors", None),
            past_key_values=past_key_values,  # passing the initilized cache
            **generate_kwargs
        )
        sequences = output.sequences[:, batch['input_tokens'].shape[1]:]
        return sequences.type(torch.int), output.past_key_values, encoded_lowcoder_states

    if FLAGS.iterative_mode:
        prefix_forward_generate = _forward_generate
        single_forward_generate = _forward_generate

    class ModelServer(LMServer):

        @staticmethod
        def loglikelihood_rolling(text):
            *output, _ = _loglikelihood_rolling(tokenizer, params, text,
                                                    func=forward_loglikelihood,
                                                    nearest_chunk_distance=FLAGS.nearest_chunk_distance,
                                                    input_length=FLAGS.input_length,
                                                    )
            return output

        @staticmethod
        def lowcoder_rolling(text, num_neighbors=2, wipe_cache=False, nearest_chunk_distance=None):
            if nearest_chunk_distance is None:
                nearest_chunk_distance = FLAGS.nearest_chunk_distance
            memory = Memory(chunk_size=64, num_neighbors=num_neighbors, nearest_chunk_distance=nearest_chunk_distance,
                            is_dense=FLAGS.dense_mem)

            for batch in rolling_iterator(tokenizer, text, FLAGS.input_length):
                outputs, past_key_values = _forward_lowcoder(params, batch)
                params.update(cache=past_key_values)
                neighbor_hidden_states, neighbor_mask, *_ = memory.add(
                    input_tokens=batch["input_tokens"],
                    encoded_hidden_states=outputs.encoded_hidden_states,
                    key_chunks=outputs.key_chunks,
                    query_chunks=outputs.query_chunks,
                )
            return neighbor_hidden_states, neighbor_mask, memory

        @staticmethod
        def lowcoder_single(text):
            batch = prefix_tokenizer(text,
                                     max_length=2 * rpt_config.chunk_size,
                                     return_tensors='np',
                                     truncation=True,
                                     padding='max_length', )
            input_mask = batch.attention_mask.astype(int)
            batch = {"input_tokens": batch.input_ids.astype(int),
                     "input_mask": input_mask}
            outputs, past_key_values = _forward_lowcoder(params, batch)
            params.update(cache=past_key_values)
            prompt_vector = outputs.encoded_hidden_states

            prompt_vector = prompt_vector.reshape([1, 1, 2 * rpt_config.chunk_size, rpt_config.hidden_size])
            prompt_mask = input_mask.reshape([1, 1, 2 * rpt_config.chunk_size])
            return torch.Tensor(prompt_vector), torch.Tensor(prompt_mask)

        @staticmethod
        def generate(text, temperature, memory_str=None, prompt=None, max_new_tokens=64, precompile=False):
            batch_size = len(text)

            n_turns = max(max_new_tokens // rpt_config.chunk_size, 1) if not precompile else 2
            if prompt is not None:
                prompt_vector, prompt_mask = ModelServer.lowcoder_single(prompt)
            if memory_str is not None and len(memory_str) > 0:
                with open(memory_str) as f:
                    memory_text = f.read()
                _, _, memory = ModelServer.lowcoder_rolling(memory_text, num_neighbors=FLAGS.num_neighbors,
                                                            nearest_chunk_distance=0)
            else:
                memory = Memory(chunk_size=64, num_neighbors=FLAGS.num_neighbors, nearest_chunk_distance=0,
                                is_dense=FLAGS.dense_mem)

            batch = prepare_prefix(prefix_tokenizer, text, FLAGS.input_length, FLAGS.add_bos_token, device)
            # TOOD: investigate this:
            # Flax RPT Retriver Encoded output
            # TODO: init_cache bruh
            outputs, past_key_values = _forward_lowcoder(hf_model, **batch)
            neighbor_hidden_states, neighbor_mask, *_ = memory.add(
                input_tokens=batch["input_tokens"],
                encoded_hidden_states=outputs.encoded_hidden_states,
                key_chunks=outputs.key_chunks,
                query_chunks=outputs.query_chunks,
                append=False,
            )

            past_key_values = None

            output_text = [tuple() for _ in range(batch_size)]

            #params.pop("cache")
            output = None
            for turn_index in range(n_turns + 1):
                if turn_index == 0:  # first iteration
                    forward_generate = prefix_forward_generate
                    chunk_index = None
                else:
                    neighbor_hidden_states, neighbor_mask, *_ = memory.add(
                        input_tokens=output,
                        encoded_hidden_states=enc_lowcoder_states.encoded_hidden_states,
                        key_chunks=enc_lowcoder_states.key_chunks,
                        query_chunks=enc_lowcoder_states.query_chunks,
                        append=False
                    )
                    if prompt is not None:
                        neighbor_hidden_states = np.concatenate([prompt_vector, neighbor_hidden_states], axis=1)
                        neighbor_mask = np.concatenate([prompt_mask, neighbor_mask], axis=1)

                    # TODO: Missing parameters
                    neighbor_hidden_states, new_past_key_values = _forward_augment(
                        hf_model,
                        hidden_states=enc_lowcoder_states.original_hidden_states,
                        neighbor_hidden_states=torch.Tensor(neighbor_hidden_states),
                        neighbor_mask=torch.Tensor(neighbor_mask),
                        past_key_values=past_key_values,
                    )
                    past_key_values = {**past_key_values, **new_past_key_values}
                    #params.update(cache=past_key_values)

                    latest_token = output[:, -1:]
                    batch.update(input_tokens=latest_token,
                                 input_mask=torch.ones_like(latest_token, dtype=torch.int32))

                    chunk_index = torch.zeros([1, 1], dtype=torch.int32)  # we are assuming batch size =1 again..
                    forward_generate = single_forward_generate

                batch.update(
                    encoded_neighbors=EncodedNeighbors(
                        neighbor_hidden_states=neighbor_hidden_states,
                        neighbor_mask=neighbor_mask,
                        chunk_index=chunk_index,
                    )
                )
                output, new_past_key_values, enc_lowcoder_states = forward_generate(batch, max_new_tokens, temperature, past_key_values=past_key_values)
                #params.update(cache=past_key_values)
                if past_key_values is not None:
                    past_key_values = {**past_key_values, **new_past_key_values}
                else:
                    past_key_values = new_past_key_values
                output_text = postproc_output(tokenizer, output, output_text, verbose=True)

            return ["".join(x) for x in output_text]

    server = ModelServer(FLAGS.lm_server)
    server.run()


if __name__ == "__main__":
    mlxu.run(main)
