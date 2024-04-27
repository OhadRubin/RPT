import json
from collections import namedtuple

import einops
import gin
import numpy as np
import optax
import torch
import transformers
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from mlxu import function_args_to_config, load_pickle, open_file
from ml_collections import ConfigDict
from typing import Optional, Tuple, Union, Dict, List
from transformers import AutoTokenizer, StoppingCriteriaList
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions

from torch_utils import make_attention_mask, make_causal_mask, combine_masks, gelu, silu, assign_slice
from dataclasses import dataclass
from transformers import LogitsProcessorList
import jax

# used just for the attention function call
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights
from EasyLM.memory_efficient_attention import dot_product_attention_multihead as efficient_dot_product_attention


RetrieverSupervision = namedtuple('RetrieverSupervision', ['nei_scores', 'nei_idx'])
EncodedNeighbors = namedtuple('EncodedNeighbors', ['neighbor_hidden_states', 'neighbor_mask',"chunk_index"])



@dataclass
class FlaxRPTRetrieverEncodedOutput:
    original_hidden_states: torch.Tensor = None
    encoded_hidden_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    key_chunks: torch.Tensor = None
    query_chunks: torch.Tensor = None
    chunk_mask: torch.Tensor = None
    preret_attention: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


@dataclass
class FlaxRPTRetrieverEncodedOutput:
    original_hidden_states: torch.Tensor = None
    encoded_hidden_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    key_chunks: torch.Tensor = None
    query_chunks: torch.Tensor = None
    chunk_mask: torch.Tensor = None
    preret_attention: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


@dataclass
class FlaxRPTRetrieverNeighborOutput:
    aux_loss: torch.Tensor = None
    loss_scale: torch.Tensor = None
    neighbor_hidden_states: torch.Tensor = None
    neighbor_mask: torch.Tensor = None
    retrieval_metrics: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class FlaxRPTLowcoderRetrieverEncodedOutput:
    hidden_states: torch.Tensor = None
    attention_mask: torch.Tensor = None
    neighbor_hidden_states: torch.Tensor = None
    neighbor_mask: torch.Tensor = None


@dataclass
class FlaxRPTModelOutput:
    last_hidden_state: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    upcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    upcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    lowcoder_last_hidden_state: Optional[torch.Tensor] = None
    lowcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    lowcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    retriever_output: FlaxRPTRetrieverNeighborOutput = None
    retriever_input: Optional[torch.Tensor] = None


@dataclass
class FlaxRPTLMOutput(transformers.utils.ModelOutput):
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    upcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    upcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    lowcoder_last_hidden_state: Optional[torch.Tensor] = None
    lowcoder_hidden_states: Optional[Tuple[torch.Tensor]] = None
    lowcoder_attentions: Optional[Tuple[torch.Tensor]] = None
    retriever_output: FlaxRPTRetrieverNeighborOutput = None
    retriever_input: Optional[torch.Tensor] = None


@dataclass
class SampleState:
    cur_len: torch.Tensor
    sequences: torch.Tensor
    running_token: torch.Tensor
    is_sent_finished: torch.Tensor
    prng_key: torch.Tensor
    model_kwargs: Dict[str, torch.Tensor]

def m1_cosine_decay_schedule(
    decay_steps: int,
    min_value:float,
    max_value:int,
    exponent: float = 1.0,
):
  if not decay_steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive decay_steps!')
  def schedule(count):
    count = torch.min(count, decay_steps)
    cosine_decay = 0.5 * (1 + torch.cos(torch.pi * count / decay_steps))
    decayed = 1-(cosine_decay ** exponent)
    decayed = (1 - min_value) * decayed + min_value
    return max_value*decayed

  return schedule

def topk_chunks(retriever_scores,num_candidates,*,where=None):
    # TODO: This used to have a @jax.vmap annotation on it, let's pytorch it
    def _topk_chunks(retriever_scores):
        return (-retriever_scores).argsort()[:num_candidates] #k = num_candidates
    if where is not None:
        retriever_scores = torch.where(where,retriever_scores,-torch.inf)
    return _topk_chunks(retriever_scores)

def create_segment_mask(total_num_chunks,n_skip_chunks):

    # TODO: This used to have a @jax.vmap annotation on it, let's pytorch it
    def _create_segment_mask(chunk_index):
        max_chunk = n_skip_chunks*(chunk_index//n_skip_chunks)
        return torch.arange(total_num_chunks)<max_chunk - 2
    return _create_segment_mask(torch.arange(total_num_chunks))


@gin.configurable
class RPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~RPTModel`]. It is used to instantiate an RPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RPT-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the RPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~RPTModel`] or [`~TFRPTModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_sequence_length (`int`, *optional*, defaults to 2048):
            Max sequence length for model (for RoPE computation)
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cca_freq (`int`, *optional*, defaults to None):
            Sets the frequency of Chunked Cross Attention projection. If None, no Chunked Cross Attention is performed.
        chunk_size (`int`, *optional*, defaults to None):
            The size of the chunks for Chunked Cross Attention.
        num_neighbors (`int`, *optional*, defaults to None):
            The number of neighbors to use for Chunked Cross Attention.
        num_sequence_chunks (`int`, *optional*, defaults to None):
            The number of chunks in max_sequence_length tokens. If None, this will be calculated as `max_sequence_length//chunk_size`.
        num_scored_neighbors (`int`, *optional*, defaults to None):
            During training this is the number of neighbors we calculate the retriever loss on.
            The model will expect to have the fields "nei_scores" and "nei_idx" in the batch with the same (...,num_sequence_chunks*num_scored_neighbors).
            If None, the retriever will not be trained.
        Example:
    ```python
    #>>> from transformers import RPTModel, RPTConfig
    #>>> # Initializing a RPT rpt-7b style configuration
    #>>> configuration = RPTConfig()
    #>>> # Initializing a model from the rpt-7b style configuration
    #>>> model = RPTModel(configuration)
    #>>> # Accessing the model configuration
    #>>> configuration = model.config
    ```"""
    model_type = "rpt"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            max_sequence_length=2048,
            document_length=16384,
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            use_cache=True,
            # pad_token_id=-1,
            bos_token_id=0,
            eos_token_id=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            tie_word_embeddings=False,
            remat_block='nothing_saveable',
            remat_attention='',
            remat_mlp='',
            scan_attention=False,
            scan_mlp=False,
            scan_query_chunk_size=1024,
            scan_key_chunk_size=2048,
            scan_mlp_chunk_size=1024,
            fcm_min_ratio=0.0,
            fcm_max_ratio=0.0,
            sliding_window: bool = True,
            n_windows: int = 1,
            window_length: int = 2048,
            cca_freq: Optional[int] = None,
            num_neighbors: Optional[int] = None,
            num_sequence_chunks: Optional[int] = None,
            chunk_size: Optional[int] = 64,
            num_scored_neighbors: Optional[int] = None,
            mesh_dim: Optional[str] = None,
            retriever_fill_value: float = -10000.0,
            threshold_nei_scores: float = 0.0,
            aux_loss_schedule_steps: Optional[int] = None,
            max_margin: Optional[float] = None,
            margin_schedule_steps: Optional[int] = None,
            ss_schedule_steps: Optional[int] = None,
            scheduled_sampling_min_prob: Optional[float] = None,
            scheduled_sampling_max_prob: Optional[float] = None,
            aux_scale: Optional[float] = None,
            augment_neighbors: bool = False,
            return_ret_metrics: bool = True,
            stride: Optional[int] = None,
            run_modules: str = "all",
            augment_across_neighbors: bool = False,
            rms_one_baseline: bool = False,
            palm_init: bool = False,
            rot_dim: Optional[int] = None,
            add_null_attn: bool = False,
            gated_ff: bool = True,
            mult_in_complex: bool = False,
            use_cca_norm2: bool = False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.document_length = document_length
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.sliding_window = sliding_window
        self.window_length = window_length
        self.n_windows = n_windows
        self.cca_freq = cca_freq
        self.chunk_size = chunk_size
        self.num_neighbors = num_neighbors
        self.aux_loss_schedule_steps = aux_loss_schedule_steps
        self.max_margin = max_margin
        self.margin_schedule_steps = margin_schedule_steps
        self.ss_schedule_steps = ss_schedule_steps
        self.scheduled_sampling_min_prob = scheduled_sampling_min_prob
        self.scheduled_sampling_max_prob = scheduled_sampling_max_prob
        self.aux_scale = aux_scale
        self.return_ret_metrics = return_ret_metrics
        # assert stride is None
        self.stride = stride
        self.run_modules = run_modules
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.augment_across_neighbors = augment_across_neighbors
        self.rms_one_baseline = rms_one_baseline
        self.add_null_attn = add_null_attn
        self.gated_ff = gated_ff
        self.mult_in_complex = mult_in_complex
        self.use_cca_norm2 = use_cca_norm2

        if num_sequence_chunks is not None:
            self.num_sequence_chunks = num_sequence_chunks
        elif max_sequence_length is not None and chunk_size is not None:
            self.num_sequence_chunks = max_sequence_length // chunk_size
        else:
            self.num_sequence_chunks = None
        self.num_document_chunks = self.document_length // chunk_size
        self.num_scored_neighbors = num_scored_neighbors
        self.mesh_dim = mesh_dim
        self.retriever_fill_value = retriever_fill_value
        self.threshold_nei_scores = threshold_nei_scores
        self.augment_neighbors = augment_neighbors
        self.palm_init = palm_init
        self.rot_dim = rot_dim
        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    @classmethod
    def get_tokenizer(cls, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                                                  pad_token='<|endoftext|>',
                                                  mask_token='<|endoftext|>',
                                                  **kwargs)
        return tokenizer

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return 'params', 'dropout', 'fcm'

    @classmethod
    def load_config(cls, path):
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['rpt_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')


class RPTRMSNorm(nn.Module):
    """
    RMS normalization layer
    """

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.eps = self.config.rms_norm_eps
        if self.config.rms_one_baseline:
            self.weight = nn.Parameter(torch.zeros(self.config.hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(self.config.hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.square(x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.promote_types(self.dtype, torch.float32))
        output = self._norm(x).type(self.dtype)
        weight = torch.asarray(self.weight, dtype=self.dtype)
        if self.config.rms_one_baseline:
            out = output * (1 - weight)
        else:
            out = output * weight

        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].type(dtype) / dim))
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).type(dtype)  # type: ignore
    sin, cos = torch.sin(freqs), torch.cos(freqs)
    freqs_cis = cos + 1j * sin
    return torch.asarray(freqs_cis)


def apply_rotary_emb_(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        freqs_cis_k: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    reshape_xq = xq.type(torch.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.type(torch.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    freqs_cis = torch.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    # freqs_cis = (1, 1024, 1, 64)
    xq_out = xq_ * freqs_cis
    xq_out = torch.stack((torch.real(xq_out), torch.imag(xq_out)), dim=-1).reshape(*xq_out.shape[:-1], -1)
    if freqs_cis_k is None:
        xk_out = xk_ * freqs_cis
        xk_out = torch.stack((torch.real(xk_out), torch.imag(xk_out)), dim=-1).reshape(*xk_out.shape[:-1], -1)
    else:
        freqs_cis_k = torch.reshape(freqs_cis_k, (*freqs_cis_k.shape[:2], 1, *freqs_cis_k.shape[2:]))
        xk_out = xk_ * freqs_cis_k
        xk_out = torch.stack((torch.real(xk_out), torch.imag(xk_out)), dim=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.type(dtype), xk_out.type(dtype)


def mult_in_complex(xq, xk):
    # Reshape input vectors to isolate real and imaginary parts
    reshape_xq = xq.reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.reshape(*xk.shape[:-1], -1, 2)

    # Extract real and imaginary parts
    real_xq = reshape_xq[..., 0]
    imag_xq = reshape_xq[..., 1]
    real_xk = reshape_xk[..., 0]
    imag_xk = reshape_xk[..., 1]

    # Perform element-wise multiplication in complex without converting to complex numbers
    real_part = real_xq * real_xk - imag_xq * imag_xk
    imag_part = real_xq * imag_xk + imag_xq * real_xk

    # Stack and reshape the result to match the desired output format
    out = torch.stack((real_part, imag_part), dim=-1).reshape(*real_part.shape[:-1], -1)

    return out


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        freqs_cis_k: torch.Tensor = None,
        rot_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rot_dim is not None and rot_dim > 0:

        # Separate the tensors based on the rotation dimensions
        xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
        xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

        # freqs_q_rot = freqs_q[..., :rot_dim]
        # freqs_k_rot = freqs_k[..., :rot_dim] if freqs_k is not None else None

        # Apply the function on the parts that need rotation
        print(freqs_cis.shape, xq_rot.shape, xk_rot.shape)
        xq_rot, xk_rot = apply_rotary_emb_(xq_rot, xk_rot, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

        # Concatenate the rotated and non-rotated parts
        xq_out = torch.concatenate((xq_rot, xq_pass), dim=-1)
        xk_out = torch.concatenate((xk_rot, xk_pass), dim=-1)
    else:
        xq_out, xk_out = apply_rotary_emb_(xq, xk, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

    return xq_out, xk_out


def repeat_kv(hidden_states, n_rep: int):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = torch.broadcast_to(hidden_states,
                                       (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RPTAttention(nn.Module):
    """
    The transformer's masked self attention layer
    """

    def __init__(self, config: RPTConfig, dtype: torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # TODO: DEVICE!!!
        self.wq = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wk = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wv = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wo = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # self.resid_dropout = nn.Dropout(rate=config.resid_pdrop,broadcast_dims=(0,))

        self.causal_mask = make_causal_mask(torch.ones((1, config.max_sequence_length), dtype=torch.bool), dtype=torch.bool)
        if self.config.rot_dim is not None and self.config.rot_dim > 0:
            rot_dim = self.config.rot_dim
        else:
            rot_dim = self.head_dim
        self.freqs_cis = precompute_freqs_cis(
            rot_dim,
            config.max_sequence_length * 2,
            dtype=self.dtype,
        )
        if self.config.add_null_attn:
            #self.null_k = self.param(f'null_k', torch.nn.init.normal(0.0001), (1, 1, self.num_heads, self.head_dim))
            #self.null_v = self.param(f'null_v', torch.nn.init.normal(0.0001), (1, 1, self.num_heads, self.head_dim))

            self.null_k = torch.nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))
            self.null_v = torch.nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def _concatenate_to_cache(self, key, value, query, attention_mask, layer_past):
        is_initialized = layer_past is not None

        if not is_initialized:
            key_max_shape = list(key.shape)
            value_max_shape = list(value.shape)
            query_max_shape = list(query.shape)
            attention_mask_max_shape = list(attention_mask.shape)
            key_max_shape[1] = value_max_shape[1] = query_max_shape[1] = attention_mask_max_shape[1] = self.config.window_length

            zero_key = torch.zeros(key_max_shape, dtype=key.dtype)
            zero_value = torch.zeros(value_max_shape, dtype=key.dtype)
            zero_query = torch.zeros(query_max_shape, dtype=key.dtype)
            zero_attention_mask = torch.zeros(attention_mask_max_shape, dtype=key.dtype)

            past_xk, past_xv, past_attention_mask = self._concatenate_to_cache(zero_key, zero_value, zero_query, zero_attention_mask, [{}])
            presets = ({'xk': past_xk, 'xv': past_xv, 'cache_mask': past_attention_mask},)
            return self._concatenate_to_cache(key, value, query, attention_mask, presets)

        past_key = layer_past[0]['xk'] if layer_past is not None and 'xk' in layer_past[0] else torch.zeros(key.shape, dtype=key.dtype)
        past_value = layer_past[0]['xv'] if layer_past is not None and 'xv' in layer_past[0] else torch.zeros(value.shape, dtype=key.dtype)
        cache_mask = layer_past[0]['cache_mask'] if layer_past is not None and 'cache_mask' in layer_past[0] else torch.zeros(attention_mask.shape, dtype=key.dtype)

        # detect if we're initializing by absence of existing cache data.
        is_initialized = past_key is not None
        cached_key = past_key
        cached_value = past_value

        # TODO: Always is bruh
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.shape
            # update key, value caches with our new 1d spatial slices
            num_updated_cache_vectors = query.shape[-3]
            shift = max_length - num_updated_cache_vectors

            def cur_index_big_torch(key, value, attention_mask):
                indices = (0,) * len(batch_dims) + (shift, 0, 0)

                key_operand = torch.roll(cached_key, shifts=-num_updated_cache_vectors, dims=-3)
                key = assign_slice(key_operand, key, indices)

                value_operand = torch.roll(cached_value, shifts=-num_updated_cache_vectors, dims=-3)
                value = assign_slice(value_operand, value, indices)

                mask_operand = torch.roll(cache_mask, shifts=-num_updated_cache_vectors, dims=-1)

                attention_mask = assign_slice(mask_operand, attention_mask.type(mask_operand.dtype), (0,) * len(batch_dims) + (shift,))

                return key, value, attention_mask

            # cond_input = (key, value, attention_mask,)
            # key, value, attention_mask = lax.cond(cur_index < max_length,
            #                                     cur_index_small,
            #                                     cur_index_big,
            #                                     *cond_input)
            # else:
            key, value, attention_mask = cur_index_big_torch(key, value, attention_mask)

            cached_key.value = key
            cached_value.value = value
            cache_mask.value = attention_mask

        return key, value, attention_mask

    # TODO: Remember you removed to position_id parameter
    def forward(
            self,
            hidden_states,
            layer_past,
            attention_mask,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask=None,
            sliding_window=False,
            disable_cache: bool = False
    ):
        n_windows = self.config.n_windows
        # stride = self.config.stride if not disable_cache else None

        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        query_length = xq.shape[-3]
        batch_size = hidden_states.shape[0]
        query_attention_mask = attention_mask

        presets = None
        if layer_past is not None or init_cache and not disable_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask, layer_past)
            presets = ({'xk': xk, 'xv': xv, 'cache_mask': attention_mask},)

        key_length = xk.shape[-3]

        position_ids = torch.broadcast_to(
            torch.clip(torch.cumsum(query_attention_mask, dim=-1) - 1, min=0),
            (batch_size, query_length)
        ).type(torch.int)

        if key_length != query_length:
            position_ids_k = torch.broadcast_to(
                torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
                (batch_size, key_length)
            ).type(torch.int)
            freqs_cis_k = np.take(self.freqs_cis, position_ids_k, axis=0)
            position_ids += position_ids_k.max() - position_ids.max()
        else:
            position_ids_k = position_ids
            freqs_cis_k = None

        if sliding_window and n_windows > 1:
            query_length = query_length // n_windows
            key_length = key_length // n_windows
            batch_size = batch_size * n_windows
            attention_mask = rearrange(attention_mask, 'b (s l) -> (b s) l', s=n_windows)

        freqs_cis = np.take(self.freqs_cis, position_ids, axis=0)


        if layer_past is not None or presets is not None:
            causal_mask = make_attention_mask(position_ids, position_ids_k, lambda x, y: x >= y,
                                              extra_batch_dims=0, dtype=torch.bool)
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        causal_mask = torch.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        attention_mask = torch.broadcast_to(attention_mask.unsqueeze(0).unsqueeze(0), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        attn_pdrop = self.config.attn_pdrop if not deterministic and self.config.attn_pdrop > 0.0 else 0

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis,
                                  freqs_cis_k=freqs_cis_k,
                                  dtype=self.dtype, rot_dim=self.config.rot_dim)


        # transform boolean mask into float mask
        attention_bias = torch.full(attention_mask.shape, torch.finfo(self.dtype).min).type(self.dtype)
        attention_bias[attention_mask > 0] = 0.0

        if self.num_key_value_groups > 1:
            xk = repeat_kv(xk, self.num_key_value_groups)
            xv = repeat_kv(xv, self.num_key_value_groups)

        # usual dot product attention
        if self.config.scan_attention:
            attn_weights = None
            attention_mask = einops.rearrange(
                combine_masks(attention_mask, fcm_mask),
                '... s q k -> ... s 1 q k'
            )

            dropout_rng = None
            if not deterministic and self.config.attn_pdrop > 0.0:
                dropout_rng = jax.random.key(0)

            attn_output = efficient_dot_product_attention(
                jnp.array(xq),
                jnp.array(xk),
                jnp.array(xv),
                bias=attention_mask,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                enable_dropout=not deterministic and self.config.attn_pdrop > 0.0,
                float32_logits=True,
                causal_mask=True,
                precision=self.precision,
                query_chunk_size=self.config.scan_query_chunk_size,
                key_chunk_size=self.config.scan_key_chunk_size,
            )

        else:
            if sliding_window and n_windows > 1:
                xq, xk, xv = map(lambda t: rearrange(t, 'b (s l) ... -> b s l ...', s=n_windows),
                                 (xq, xk, xv))
                attention_bias = rearrange(attention_bias, '(b s) ... -> b s ...', s=n_windows)

                past_xk, past_xv, past_attention_bias = map(lambda t: t[:, :-1, :, :, :],
                                                            (xk, xv, attention_bias))
                past_xk, past_xv = map(lambda t: torch.pad(t, [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)]),
                                       (past_xk, past_xv))
                past_attention_bias = torch.pad(past_attention_bias, [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)],
                                              constant_values=torch.finfo(past_attention_bias.dtype).min)
                xk = torch.concatenate((past_xk, xk), axis=-3)
                xv = torch.concatenate((past_xv, xv), axis=-3)
                attention_bias = torch.concatenate((past_attention_bias, attention_bias), axis=-1)

            if self.config.add_null_attn:
                xv, xk, attention_bias = self.concat_null_kv(xv, xk, attention_bias)

            # xq = torch.Size([1, 1024, 32, 128])
            # xk = torch.Size([1, 1024, 32, 128])
            # xv = torch.Size([1, 1024, 32, 128])
            # attention_mask = torch.Size([1, 1, 1024, 1024])
            # attn_pdrop = 0

            dropout_rng = None
            if not deterministic and self.config.attn_pdrop > 0.0:
                dropout_rng = jax.random.key(0)

            attn_weights = dot_product_attention_weights(
                jnp.array(xq.detach().numpy()),
                jnp.array(xk.detach().numpy()),
                bias=jnp.array(attention_bias.detach().numpy()),
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                deterministic=deterministic,
                dtype=jnp.float32,
            )

            attn_weights = torch.Tensor(attn_weights.tolist())
            print(f"{attn_weights.shape=}")
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv)
            if sliding_window and n_windows > 1:
                attn_output = rearrange(attn_output,
                                        'b s l ... -> b (s l) ...', s=n_windows)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)

        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, presets)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def concat_null_kv(self, xv, xk, attention_bias):
        attention_bias_shape = torch.Tensor(np.array(attention_bias.shape)).type(torch.int)
        attention_bias_shape[-1] = 1
        xk_shape = torch.Tensor(np.array(xk.shape)).type(torch.int)
        xk_shape[-3] = 1

        null_k = torch.broadcast_to(self.null_k, tuple(xk_shape))
        null_v = torch.broadcast_to(self.null_v, tuple(xk_shape))
        # xk = torch.Size([1, 1024, 16, 128]), null_k=torch.Size([1, 1, 16, 128])
        # xv = torch.Size([1, 1024, 16, 128]), null_v=torch.Size([1, 1, 16, 128])
        xk = torch.concatenate((xk, null_k), dim=-3) # torch.Size([1, 1025, 16, 128])
        xv = torch.concatenate((xv, null_v), dim=-3) # torch.Size([1, 1025, 16, 128])
        attention_bias = torch.concatenate((attention_bias, torch.full(tuple(attention_bias_shape), 0.0)), dim=-1) # add last dim (embedding?)
        return xv, xk, attention_bias


# TODO: Currently unused!! make sure you use it or get rid of it
def dense_init(input_tensor, config, is_embedding=False):
    if config.palm_init:
        if is_embedding:
            return torch.nn.init.normal(tensor=input_tensor, std=1.0)
        # TODO: The use of len needs more examination
        return torch.nn.init.normal(tensor=input_tensor, std=torch.sqrt(config.initializer_range / len(input_tensor)))
    else:
        return torch.nn.init.normal(tensor=input_tensor, std=config.initializer_range)


class RPTCrossAttention(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wk = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wv = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        self.wo = torch.nn.Linear(
            self.embed_dim,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype
        )

        # self.resid_dropout = nn.Dropout(rate=config.resid_pdrop,broadcast_dims=(0,))

        if self.config.rot_dim is not None and self.config.rot_dim > 0:
            rot_dim = self.config.rot_dim
        else:
            rot_dim = self.head_dim

        self.freqs_cis = precompute_freqs_cis(
            rot_dim,
            config.max_sequence_length * 2,
            dtype=self.dtype,
        )
        self.null_k = torch.nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))
        self.null_v = torch.nn.Parameter(torch.normal(mean=0, std=0.0001, size=(1, 1, self.num_heads, self.head_dim)))


    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            kv_position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
    ) -> Tuple[torch.Tensor]:

        is_cross_attention = key_value_states is not None

        if not is_cross_attention:
            xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        else:
            xq, xk, xv = self.wq(hidden_states), self.wk(key_value_states), self.wv(key_value_states)

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        query_length, key_length = xq.shape[1], xk.shape[1]
        batch_size = hidden_states.shape[0]

        if position_ids is None:
            position_ids = torch.arange(query_length, dtype=torch.int32)
            position_ids = torch.broadcast_to(position_ids[None, :], (batch_size, query_length))

        # TODO: get rid of numpy
        freqs_cis = np.take(self.freqs_cis, position_ids, axis=0)
        if not is_cross_attention:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype, rot_dim=self.config.rot_dim)
        else:
            if kv_position_ids is None:
                kv_position_ids = torch.arange(key_length, dtype=torch.int32)
                kv_position_ids = torch.broadcast_to(kv_position_ids[None, :], (batch_size, key_length))
            # TODO: Get rid of numpy
            freqs_cis_k = np.take(self.freqs_cis, kv_position_ids, axis=0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k, dtype=self.dtype,
                                      rot_dim=self.config.rot_dim)

        null_k = torch.broadcast_to(self.null_k, (batch_size, 1, self.num_heads, self.head_dim))
        xk = torch.concatenate((xk, null_k), dim=-3)

        null_v = torch.broadcast_to(self.null_v, (batch_size, 1, self.num_heads, self.head_dim))
        xv = torch.concatenate((xv, null_v), dim=-3)

        if attention_mask is not None:
            null_mask = torch.ones((attention_mask.shape[0], 1), dtype=torch.float32)
            attention_mask = torch.concatenate((attention_mask, null_mask), dim=-1)
            attention_mask = attention_mask.unsqueeze(-2).unsqueeze(-2)

            # attention_mask.shape = (16, 1, 1, 65)
            attention_bias = torch.full(attention_mask.shape, torch.finfo(self.dtype).min).type(self.dtype)
            attention_bias[attention_mask > 0] = 0.0
        else:
            attention_bias = None

        # TODO: Get rid of JAX
        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            jnp.array(xq.detach().numpy()),
            jnp.array(xk.detach().numpy()),
            bias=None if attention_bias is None else jnp.array(attention_bias.detach().numpy()),
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=jnp.float32,
            precision=None,
        )
        attn_weights = torch.Tensor(attn_weights.tolist())

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv)

        attn_output = self._merge_heads(attn_output)

        attn_output = self.wo(attn_output)

        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class RPTMLP(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype

        # TODO: Don't forget to initialize
        self.w1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size if config.gated_ff else 4 * config.hidden_size,
            dtype=self.dtype,
            bias=False,
        )
        self.w2 = nn.Linear(
            in_features=self.w1.out_features,
            out_features=config.hidden_size,
            dtype=self.dtype,
            bias=False,
        )
        if self.config.gated_ff:
            self.w3 = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.intermediate_size,
                dtype=self.dtype,
                bias=False,
            )

        # TODO: Deterministic dropout
        self.dropout = nn.Dropout(p=self.config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.gated_ff:
            x1 = silu(self.w1(x))
            x3 = self.w3(x)
            if self.config.mult_in_complex:
                x = mult_in_complex(x1, x3)
            else:
                x = x1 * x3
            x = self.w2(x)

            x = self.dropout(x)
        else:
            x = gelu(self.w1(x))
            x = self.dropout(x)
            x = self.w2(x)

        return x


class RPTLowcoderLayer(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.attention = RPTAttention(
            self.config,
            dtype=self.dtype,
        )
        self.feed_forward = RPTMLP(
            self.config,
            dtype=self.dtype
        )
        self.attention_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        self.ffn_norm = RPTRMSNorm(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask: Optional[torch.Tensor] = None,
    ):
        # run self attention on the hidden states
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            layer_past,
            attention_mask,
            deterministic,
            init_cache, # TODO: caching and stuff
            output_attentions,
            fcm_mask,
            sliding_window=self.config.sliding_window,
        )

        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        outputs = attn_outputs[1:]

        # nomalize hidden states
        feed_forward_input = self.ffn_norm(hidden_states)

        # TODO: is this syntax or is it real?
        # run the nomlaized hidden states into the MLP
        if self.config.scan_mlp:
            feed_forward_input = einops.rearrange(
                feed_forward_input,
                '... (b s) d -> ... b s d',
                b=self.config.scan_mlp_chunk_size
            )

            def mlp_forward(mlp, carry, x):
                return None, mlp(x, deterministic)

            scan_axis = feed_forward_input.ndim - 3

            _, feed_forward_hidden_states = nn.scan(
                mlp_forward,
                variable_broadcast="params",
                split_rngs={"params": False, "dropout": True},
                in_axes=scan_axis,
                out_axes=scan_axis,
            )(self.feed_forward, None, feed_forward_input)
            feed_forward_hidden_states = einops.rearrange(
                feed_forward_hidden_states,
                '... b s d -> ... (b s) d'
            )
        else:
            feed_forward_hidden_states = self.feed_forward(feed_forward_input)

        hidden_states = hidden_states + feed_forward_hidden_states

        if init_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class RPTLowcoderLayerCollection(nn.ModuleList):
    """
    Basic series of masked attention encoders
    """

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        assert (self.config.num_hidden_layers % 2) == 0, f"config.num_hidden_layers should be devisible by 2"
        num_hidden_layers = self.config.num_hidden_layers // 2
        print("In Lowcoder: Using {} layers".format(num_hidden_layers))
        self.blocks = nn.ModuleList([
            RPTLowcoderLayer(
                self.config,
                dtype=self.dtype
            ) for i in range(num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        fcm_mask = None

        presents = () if init_cache else None

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                layer_past,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
            )
            # TODO: Is this not useful?
            hidden_states = layer_outputs[0]

            if init_cache:
                presents = presents + (layer_outputs[1],)

            # TODO: Is this not useful?
            if output_attentions:
                all_attentions += (layer_outputs[2 if init_cache else 1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_values = {'lowcoder': presents}
        outputs = (hidden_states, past_key_values, all_hidden_states, all_attentions)

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )


class RPTLowcoder(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.layers = RPTLowcoderLayerCollection(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        outputs = self.layers(
            hidden_states,
            past_key_values,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values
        )


class RPTChunkedCrossAttention(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.chunk_size = self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.cross_attention = RPTCrossAttention(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            neighbor_hidden_states,
            neighbor_mask,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            deterministic: bool = True,
    ) -> Tuple[torch.Tensor]:

        chunk_size = self.chunk_size
        causal_padding = chunk_size - 1
        num_devices, seq_len, hidden_dim = hidden_states.shape
        num_document_chunks, num_neighbors, _, _ = neighbor_hidden_states.shape

        # -> (-1, num_devices_chunks, num_neighbors, 2*chunk_size, hidden_dim)
        neighbor_hidden_states = neighbor_hidden_states.reshape([-1, 2 * chunk_size * num_neighbors, hidden_dim])
        neighbor_mask = neighbor_mask.reshape([-1, 2 * chunk_size * num_neighbors])
        local_device_count = hidden_states.shape[0]
        if num_document_chunks > 1:
            num_devices_chunks = num_document_chunks // local_device_count
            # ->  (-1 ,chunk_size, hidden_dim)
            hidden_states = hidden_states.reshape([-1, num_devices_chunks * chunk_size, hidden_dim])
            # TODO: remove numpy
            hidden_states = torch.Tensor(np.pad(hidden_states[:,causal_padding:,:], ((0,0),(0, causal_padding),(0,0)), 'constant'))
            hidden_states = hidden_states.reshape([-1, chunk_size, hidden_dim])

            position_ids = torch.arange(chunk_size) + chunk_size - 1
            position_ids = torch.broadcast_to(position_ids[None, :], (hidden_states.shape[0], chunk_size))
        else:
            hidden_states = hidden_states.reshape([1, 1, hidden_dim])
            assert position_ids is not None

        kv_position_ids = torch.arange(2 * chunk_size)
        kv_position_ids = torch.broadcast_to(kv_position_ids[None, :],
                                           (num_document_chunks * num_neighbors, 2 * chunk_size))
        kv_position_ids = kv_position_ids.reshape([-1, 2 * chunk_size * num_neighbors])

        # cross attention
        output = self.cross_attention(
            hidden_states=hidden_states,
            key_value_states=neighbor_hidden_states,
            position_ids=position_ids,
            kv_position_ids=kv_position_ids,
            attention_mask=neighbor_mask,
            output_attentions=output_attentions,
            deterministic=deterministic)

        # reshape back to original sequence
        cross_attention_out = output[0]
        if num_document_chunks > 1:
            cross_attention_out = cross_attention_out.reshape([-1, num_devices_chunks * chunk_size, hidden_dim])
            # # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
            # TODO: numpy
            cross_attention_out = torch.Tensor(np.pad(cross_attention_out, ((0, 0), (causal_padding, 0), (0, 0)), 'constant')[:,
                                  :-causal_padding])
        cross_attention_out = cross_attention_out.reshape([num_devices, seq_len, hidden_dim])
        return (cross_attention_out,) + output[1:]


class RPTUpcoderLayer(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, has_cca: bool = False):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.has_cca = has_cca
        if self.has_cca:
            self.cca = RPTChunkedCrossAttention(
                self.config,
                dtype=self.dtype
            )
            self.cca_norm = RPTRMSNorm(self.config, dtype=self.dtype)
            if self.config.use_cca_norm2:
                self.cca_norm2 = RPTRMSNorm(self.config, dtype=self.dtype)
            else:
                self.cca_norm2 = None

        else:
            self.cca = None
            self.cca_norm = None

        self.attention = RPTAttention(
            self.config,
            dtype=self.dtype
        )

        self.feed_forward = RPTMLP(
            self.config,
            dtype=self.dtype
        )
        self.attention_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        self.ffn_norm = RPTRMSNorm(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            position_ids=None,
            neighbor_hidden_states=None,
            neighbor_mask=None,
            chunk_index=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            fcm_mask: Optional[torch.Tensor] = None,
    ):

        print(f"In Upcoder Layer: Using CCA: {self.has_cca}")
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            layer_past,
            attention_mask,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
            sliding_window=self.config.sliding_window,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        outputs = attn_outputs[1:]

        if self.cca is not None and neighbor_hidden_states is not None:
            if self.cca_norm2 is not None:
                neighbor_hidden_states = self.cca_norm2(neighbor_hidden_states)

            cca_output = self.cca(hidden_states=self.cca_norm(hidden_states),
                                  neighbor_hidden_states=neighbor_hidden_states,
                                  neighbor_mask=neighbor_mask,
                                  position_ids=chunk_index,
                                  output_attentions=output_attentions,
                                  deterministic=deterministic,

                                  )
            cca_hidden_states = cca_output[0]
            hidden_states = cca_hidden_states + hidden_states

        feed_forward_input = self.ffn_norm(hidden_states)

        if self.config.scan_mlp:
            feed_forward_input = einops.rearrange(
                feed_forward_input,
                '... (b s) d -> ... b s d',
                b=self.config.scan_mlp_chunk_size
            )

            def mlp_forward(mlp, carry, x):
                return None, mlp(x, deterministic)

            scan_axis = feed_forward_input.ndim - 3

            # TODO: handle
            _, feed_forward_hidden_states = nn.scan(
                mlp_forward,
                variable_broadcast="params",
                split_rngs={"params": False, "dropout": True},
                in_axes=scan_axis,
                out_axes=scan_axis,
            )(self.feed_forward, None, feed_forward_input)
            feed_forward_hidden_states = einops.rearrange(
                feed_forward_hidden_states,
                '... b s d -> ... (b s) d'
            )
        else:
            feed_forward_hidden_states = self.feed_forward(
                feed_forward_input,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        if init_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class RPTUpcoderLayerCollection(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        assert (self.config.num_hidden_layers % 2) == 0, f"config.num_hidden_layers should be divisible by 2"
        num_hidden_layers = self.config.num_hidden_layers // 2
        print("In Upcoder: Using {} layers".format(num_hidden_layers))

        def has_cca(layer_index):
            if self.config.cca_freq is None or self.config.cca_freq == 0:
                return False
            return (layer_index % self.config.cca_freq) == 0

        self.blocks = nn.ModuleList([
            RPTUpcoderLayer(
                self.config,
                dtype=self.dtype,
                has_cca=has_cca(i),
            ) for i in range(num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            neighbor_hidden_states=None,
            neighbor_mask=None,
            chunk_index=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        presents = () if init_cache else None

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.blocks))

        for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,  # 0
                layer_past,
                attention_mask,  # 1
                position_ids,  # 2
                neighbor_hidden_states,  # 3
                neighbor_mask,  # 4
                chunk_index,  # 5
                deterministic,  # 6
                init_cache,  # 7
                output_attentions,  # 8
                None,  # fcm_mask= #9
            )
            hidden_states = layer_outputs[0]

            if init_cache:
                presents = presents + (layer_outputs[1],)

            if output_attentions:
                all_attentions += (layer_outputs[2 if init_cache else 1],)
                if block.has_cca:
                    all_cross_attentions += (layer_outputs[3 if init_cache else 2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return outputs + (presents,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            past_key_values=presents,
        )


class RPTNeighborAugmentor(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.postret_bidir_attention = RPTCrossAttention(self.config, dtype=self.dtype)
        self.postret_bi_attention_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        self.query_nei_xattention_qnorm = RPTRMSNorm(self.config, dtype=self.dtype)
        self.query_nei_xattention_knorm = RPTRMSNorm(self.config, dtype=self.dtype)

        self.query_nei_xattention = RPTCrossAttention(self.config, dtype=self.dtype)

    def forward(self, hidden_states: torch.Tensor, neighbor_hidden_states: torch.Tensor, neighbor_mask: torch.Tensor, output_attentions: torch.Tensor, deterministic: bool,
                 query_hidden_states: torch.Tensor = None):
        neighbor_hidden_states_shape = neighbor_hidden_states.shape
        num_document_chunks, num_neighbors, ret_size, hidden_dim = neighbor_hidden_states_shape
        assert ret_size == 2 * self.config.chunk_size
        neighbor_hidden_states = neighbor_hidden_states.reshape([num_document_chunks * num_neighbors,
                                                                 ret_size,
                                                                 hidden_dim])
        neighbor_mask = neighbor_mask.reshape([num_document_chunks * num_neighbors, ret_size])

        # Non-causal self attention with the two parts of the neighbor chunk
        # For each neighbor we also retrieve it's subsequent chunk, so it helps to have the two parts of the neighbor chunk
        # Attend to each other (it was only causal before)
        postret_bi_output = self.postret_bidir_attention(
            self.postret_bi_attention_norm(neighbor_hidden_states),
            attention_mask=neighbor_mask,
            deterministic=deterministic,
            output_attentions=output_attentions)  # need(!!!) to cache this during generation
        neighbor_hidden_states = postret_bi_output[0] + neighbor_hidden_states

        # Cross Attention with the neighbor chunk hidden states as Q, and the query chunk hidden states as KV
        if query_hidden_states is None:  # if we didn't get it from update_inputs_for_generation
            query_hidden_states = hidden_states
        query_hidden_states = einops.repeat(query_hidden_states.reshape([-1, self.config.chunk_size, hidden_dim]),
                                            'b n d -> (b k) n d', n=self.config.chunk_size, k=num_neighbors)
        assert query_hidden_states.shape[0] == num_document_chunks * num_neighbors

        augmented_xattention_output = self.query_nei_xattention(
            hidden_states=self.query_nei_xattention_qnorm(neighbor_hidden_states),
            key_value_states=self.query_nei_xattention_knorm(query_hidden_states),
            output_attentions=output_attentions,
            deterministic=deterministic
        )
        neighbor_hidden_states = augmented_xattention_output[0] + neighbor_hidden_states

        neighbor_hidden_states = neighbor_hidden_states.reshape(neighbor_hidden_states_shape)

        return (neighbor_hidden_states,) + postret_bi_output[1:] + augmented_xattention_output[1:]


class RPTCrossNeighborAugmentor(nn.Module):
    def __init__(self, config: RPTConfig, device_count, num_devices_chunks, num_neighbors, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.cross_neig_causal_att = RPTAttention(self.config, dtype=self.dtype)
        self.xnei_norm1 = RPTRMSNorm(self.config, dtype=self.dtype)
        self.xnei_norm2 = RPTRMSNorm(self.config, dtype=self.dtype)
        # [device_count, num_devices_chunks*num_neighbors, 1]
        # TODO: handle initialization with dense init
        # TODO: missing in_features, need to run to figure out
        self.weight = nn.Linear(in_features=self.config.hidden_size, out_features=1, dtype=self.dtype, bias=True)
        if self.config.use_xnei_bias:
            self.xnei_bias = torch.nn.Parameter(torch.normal(mean=0, std=0.01, size=(1, 1, 1, self.config.hidden_size)))
        else:
            self.xnei_bias = None

    def forward(self,
                 neighbor_hidden_states: torch.Tensor,
                 layer_past,
                 neighbor_mask: torch.Tensor,
                 output_attentions: torch.Tensor,
                 init_cache,
                 deterministic: bool,
                 device_count: int):
        num_document_chunks, num_neighbors, ret_size, hidden_dim = neighbor_hidden_states.shape
        if num_document_chunks > 1:
            num_devices_chunks = num_document_chunks // device_count
            # this pooled tensor has shape [device_count, num_devices_chunks*num_neighbors, hidden_dim]
        else:
            num_devices_chunks = 1
        new_shape = [device_count, num_devices_chunks * num_neighbors, ret_size, hidden_dim]
        pooled_neighbor_hidden_states = neighbor_hidden_states.reshape(new_shape).mean(dim=-2)

        pooled_neighbor_mask = neighbor_mask.reshape([device_count, num_devices_chunks * num_neighbors, ret_size]).any(dim=-1)

        cross_neig_out = self.cross_neig_causal_att(
            hidden_states=self.xnei_norm1(pooled_neighbor_hidden_states),
            layer_past=layer_past,
            attention_mask=pooled_neighbor_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
            init_cache=init_cache,
            sliding_window=False,
            disable_cache=False,
        )

        pooled_neighbor_hidden_states = cross_neig_out[0] + pooled_neighbor_hidden_states
        pooled_neighbor_hidden_states = self.xnei_norm2(pooled_neighbor_hidden_states)
        ret_gate_score = self.weight(
            pooled_neighbor_hidden_states)  # [device_count, num_devices_chunks*num_neighbors, 1]
        ret_gate_score = rearrange(ret_gate_score, 't (c k) 1 -> (t c) k 1 1 ',
                                   t=device_count, c=num_devices_chunks, k=num_neighbors)

        if self.xnei_bias is not None:
            neighbor_hidden_states += ret_gate_score * self.xnei_bias
        else:
            ret_gate_score = 0.1 + 0.9 * torch.sigmoid(ret_gate_score / hidden_dim)
            neighbor_hidden_states = ret_gate_score * neighbor_hidden_states


        if init_cache:
            outputs = (neighbor_hidden_states, cross_neig_out[1]) + cross_neig_out[1:]
        else:
            outputs = (neighbor_hidden_states,) + cross_neig_out[1:]

        return outputs


class RPTUpcoder(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype

        if self.config.augment_neighbors:
            self.neighbor_augmentor = RPTNeighborAugmentor(self.config, dtype=self.dtype)
        else:
            self.neighbor_augmentor = None

        if self.config.augment_across_neighbors:
            # TODO: Fix parameters
            self.neighbor_cross_augmentor = RPTCrossNeighborAugmentor(self.config, device_count=1, dtype=self.dtype, num_neighbors=self.config.num_neighbors, num_devices_chunks=self.config.num_document_chunks)
        else:
            self.neighbor_cross_augmentor = None
        self.layers = RPTUpcoderLayerCollection(self.config, dtype=self.dtype)

    def augment(self,
                hidden_states: torch.Tensor,
                neighbor_hidden_states: torch.Tensor,
                neighbor_mask: torch.Tensor,
                deterministic: bool = True,
                output_attentions: bool = False,
                init_cache=False,
                layer_past=None,
                ):
        if self.neighbor_augmentor is not None and neighbor_hidden_states is not None:
            nei_aug_outputs = self.neighbor_augmentor(
                hidden_states,
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            neighbor_hidden_states = nei_aug_outputs[0]
            past_key_values = None
        if self.neighbor_cross_augmentor is not None and neighbor_hidden_states is not None:
            nei_xaug_outputs = self.neighbor_cross_augmentor(
                neighbor_hidden_states=neighbor_hidden_states,
                layer_past=layer_past,
                neighbor_mask=neighbor_mask,
                init_cache=init_cache,
                deterministic=deterministic,
                output_attentions=output_attentions,
                device_count=hidden_states.shape[0],
            )
            neighbor_hidden_states = nei_xaug_outputs[0]
            past_key_values = nei_xaug_outputs[1]

        # TODO: Change var name
        return neighbor_hidden_states, past_key_values


    def forward(
            self,
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            neighbor_hidden_states=None,
            neighbor_mask=None,
            chunk_index=None,
            deterministic: bool = True,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        augment_cache = past_key_values['augment'] if past_key_values is not None and 'augment' in past_key_values else None
        upcoder_cache = past_key_values['upcoder'] if past_key_values is not None and 'upcoder' in past_key_values else None

        agument_past_key_values = None
        if chunk_index is None:
            augmented_outputs = self.augment(
                hidden_states,
                neighbor_hidden_states,
                neighbor_mask,
                deterministic,
                output_attentions,
                init_cache=init_cache,
                layer_past=augment_cache,
            )

            neighbor_hidden_states = augmented_outputs[0]
            agument_past_key_values = augmented_outputs[1]
        # else We are generating... And have already augmented the neighbor hidden states.

        outputs = self.layers(
            hidden_states,
            upcoder_cache,
            attention_mask,
            position_ids=position_ids,
            neighbor_hidden_states=neighbor_hidden_states,
            neighbor_mask=neighbor_mask,
            chunk_index=chunk_index,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: Cringe
        outputs.past_key_values = {
            'upcoder': outputs.past_key_values,
            'augment': agument_past_key_values,
        }

        if not return_dict:
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )


class RPTRetriever(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.preret_bidir_attention = RPTCrossAttention(self.config, dtype=self.dtype)
        self.preret_bi_attention_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        self.pre_key_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        # TODO: handle initialization
        # TODO: handle input size
        self.key_projection = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.dtype,
            bias=True
        )
        self.pre_query_norm = RPTRMSNorm(self.config, dtype=self.dtype)
        # TODO: handle initialization
        # TODO: handle input size
        self.query_projection = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            dtype=self.dtype,
            bias=True,
        )
        self.num_neighbors = self.config.num_neighbors


    def preret_encode(self,
                      hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor,
                      deterministic: bool,
                      output_attentions: bool = False, ):
        original_hidden_states_shape = hidden_states.shape
        original_attention_mask_shape = attention_mask.shape

        # TODO: verify equivilance
        original_hidden_states = einops.rearrange(hidden_states, 'b (l c) ... -> (b l) c ... ', c=self.config.chunk_size)
        attention_mask = einops.rearrange(attention_mask, 'b (l c) ... -> (b l) c ... ', c=self.config.chunk_size)

        # add a chunk dimension
        # 1. apply bi-dir attention
        preret_bi_output = self.preret_bidir_attention(
            self.preret_bi_attention_norm(original_hidden_states),
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions)
        encoded_hidden_states = preret_bi_output[0] + original_hidden_states

        # 2. pool
        pooled_hidden_states = encoded_hidden_states.mean(dim=-2)

        # 3. project to query chunks and key chunks
        key_chunks = self.key_projection(self.pre_key_norm(pooled_hidden_states))
        query_chunks = self.query_projection(self.pre_query_norm(pooled_hidden_states))
        chunk_mask = attention_mask.type(torch.bool).any(-1)[..., None]
        original_hidden_states = original_hidden_states.reshape(original_hidden_states_shape)
        attention_mask = attention_mask.reshape(original_attention_mask_shape)

        return FlaxRPTRetrieverEncodedOutput(
            original_hidden_states=original_hidden_states,
            encoded_hidden_states=encoded_hidden_states,
            attention_mask=attention_mask,
            key_chunks=key_chunks,
            query_chunks=query_chunks,
            chunk_mask=chunk_mask,
            preret_attention=preret_bi_output[1:])



# TODO: Figure out the PyTorch version

class RPTModule(nn.Module):
    base_model_prefix = 'transformer'

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        # the embedding dim
        self.embed_dim = self.config.hidden_size

        # TODO: Comon size
        self.cached_array = torch.zeros(size=(1, config.chunk_size, config.hidden_size), dtype=self.dtype)

        # TODO: move this wte and dropout into the lowcoder.

        # define a dropout layer
        self.dropout = nn.Dropout(p=self.config.embd_pdrop)

        # TODO: handle init
        # word to embedding module (layer)
        self.wte = nn.Embedding(
            self.config.vocab_size,  # input size
            self.config.hidden_size,  # embedding size
            dtype=self.dtype
        )

        self.lowcoder = RPTLowcoder(self.config, dtype=self.dtype)
        if self.config.cca_freq is not None and self.config.cca_freq > 0:
            self.retriever = RPTRetriever(self.config, dtype=self.dtype)
        else:
            self.retriever = None

        self.upcoder = RPTUpcoder(self.config, dtype=self.dtype)

        # TODO: move this ln_f into the upcoder.
        self.ln_f = RPTRMSNorm(self.config, dtype=self.dtype)

    # TODO: Verify correctness
    def _concatenate_to_lowcoder_cache(self, array):
        *batch_dims, _, hidden_dim = array.shape
        chunk_size = self.config.chunk_size

        last_chunk = array[..., -chunk_size:, :]

        num_updated_cache_vectors = last_chunk.shape[-2]
        shift = self.config.chunk_size - num_updated_cache_vectors  # will need to update if I change retrieval stride
        indices = tuple([slice(None)] * len(batch_dims)) + (slice(shift, None), slice(None))

        self.cached_array = torch.roll(self.cached_array, shifts=-num_updated_cache_vectors, dims=-2)
        self.cached_array[indices] = last_chunk


    def forward(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            position_ids,
            deterministic=True,
            retriever_supervision: RetrieverSupervision = None,
            train_step: Optional[int] = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            upcoder_input=None,
            encoded_neighbors: Optional[EncodedNeighbors] = None,
    ):
        lowcoder_outputs = None
        retriever_output = None
        neighbor_hidden_states = None
        neighbor_mask = None
        chunk_index = None
        retriever_input = None

        if upcoder_input is None:

            input_embeds = self.wte(input_ids)

            # TODO: Determinsitc
            hidden_states = self.dropout(input_embeds)

            lowcoder_outputs = self.lowcoder(
                hidden_states,
                past_key_values['lowcoder'],
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = lowcoder_outputs.last_hidden_state if return_dict else lowcoder_outputs[0]
            self._concatenate_to_lowcoder_cache(hidden_states)

            retriever_input = hidden_states
            if self.retriever is not None:
                if encoded_neighbors is not None:
                    neighbor_hidden_states = encoded_neighbors.neighbor_hidden_states
                    neighbor_mask = encoded_neighbors.neighbor_mask
                    chunk_index = encoded_neighbors.chunk_index
                else:
                    retriever_output = self.retriever(hidden_states=retriever_input,
                                                      attention_mask=attention_mask,
                                                      retriever_supervision=retriever_supervision,
                                                      deterministic=deterministic,
                                                      train_step=train_step)
                    neighbor_hidden_states = retriever_output.neighbor_hidden_states
                    neighbor_mask = retriever_output.neighbor_mask
        else:
            hidden_states = upcoder_input.hidden_states
            attention_mask = upcoder_input.attention_mask
            neighbor_hidden_states = upcoder_input.neighbor_hidden_states
            neighbor_mask = upcoder_input.neighbor_mask

        upcoder_outputs = self.upcoder(
            hidden_states,
            past_key_values,
            attention_mask,
            position_ids=position_ids,
            neighbor_hidden_states=torch.Tensor(neighbor_hidden_states),
            neighbor_mask=torch.Tensor(neighbor_mask),
            chunk_index=chunk_index,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = upcoder_outputs.last_hidden_state if return_dict else upcoder_outputs[0]
        hidden_states = self.ln_f(hidden_states)

        past_key_values = {**lowcoder_outputs.past_key_values, **upcoder_outputs.past_key_values}

        if not return_dict:
            return (hidden_states,) + upcoder_outputs + lowcoder_outputs, past_key_values # TODO: Cringe

        return FlaxRPTModelOutput(
            past_key_values=past_key_values,
            last_hidden_state=upcoder_outputs.last_hidden_state,
            upcoder_hidden_states=upcoder_outputs.hidden_states,
            upcoder_attentions=upcoder_outputs.attentions,
            cross_attentions=None,
            lowcoder_last_hidden_state=lowcoder_outputs.last_hidden_state if lowcoder_outputs is not None else None,
            lowcoder_hidden_states=lowcoder_outputs.hidden_states if lowcoder_outputs is not None else None,
            lowcoder_attentions=lowcoder_outputs.attentions if lowcoder_outputs is not None else None,
            retriever_output=retriever_output,
            retriever_input=retriever_input,
        )


# from transformers.modeling_flax_utils import
from transformers.generation.logits_process import LogitsProcessorList

from transformers.modeling_utils import PreTrainedModel

class RPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "transformer"
    config_class = RPTConfig
    module_class: nn.Module = None

    def __init__(self, config: RPTConfig, input_shape=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.module = self.module_class(config=config, **kwargs)
        self.input_shape = input_shape

    def init_weights(self):
        pass

    def forward(
            self,
            input_ids,
            attention_mask=None,
            params: dict = None,
            past_key_values: dict = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            encoded_neighbors: Optional[EncodedNeighbors] = None,

    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is not None:
            attention_mask = torch.Tensor(attention_mask).type(dtype=torch.int)

        outputs = self.module(
            input_ids=torch.Tensor(input_ids).type(torch.int),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            encoded_neighbors=encoded_neighbors,
            deterministic=True,
            retriever_supervision=None,
            train_step=None,
            init_cache=True, # TODO: FALSE!!!
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: Caching

        return outputs

    def preret_forward(
            self,
            hidden_states,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            output_attentions: bool = False,
    ):

        apply_kwargs = self.create_apply_kwargs(params)

        outputs = self.module._encode_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=not train,
            output_attentions=output_attentions,
            **apply_kwargs
        )

        return outputs

    def augment_forward(
            self,
            hidden_states,
            neighbor_hidden_states,
            neighbor_mask,
            past_key_values: dict = None,
            params: dict = None,
            train: bool = False,
            output_attentions: bool = False
    ):

        apply_kwargs = self.create_apply_kwargs(params, past_key_values)

        outputs, past_key_values = self.module.apply(
            hidden_states=hidden_states,
            neighbor_hidden_states=neighbor_hidden_states,
            neighbor_mask=neighbor_mask,
            deterministic=not train,
            output_attentions=output_attentions,
            method=self.module._augment_forward,
            **apply_kwargs
        )

        return outputs, past_key_values.get("intermediates", None)

    def lowcoder_forward(
            self,
            input_ids,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            past_key_values: dict = None,
            output_attentions: bool = False
    ):

        apply_kwargs = self.create_apply_kwargs(params, past_key_values)

        outputs, past_key_values = self.module.apply(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=not train,
            output_attentions=output_attentions,
            method=self.module._lowcoder_forward,
            **apply_kwargs
        )
        return outputs, past_key_values.get("intermediates", None)

    def create_apply_kwargs(self, params, past_key_values=None):
        return {}

    # TODO: Cringe
    def pad_to_closest_multiple_of_k(self, arr, k):
        length = arr.size(1)
        remainder = length % k
        if remainder == 0:
            return arr
        else:
            pad_length = k - remainder
            padding = torch.zeros(pad_length, dtype=arr.dtype, device=arr.device)
            padded_arr = torch.cat((padding.unsqueeze(0), arr), dim=1)
            return padded_arr


    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ):
        # init values
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = input_ids

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            has_reached_max_length = state.cur_len % 64 == 0
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits)

            # 5295
            # Array([4146024105,  967050713], dtype=uint32) -> 1383
            # 8 -> 8.617962
            next_token = jax.random.categorical(prng_key, logits.numpy(), axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = torch.concatenate([state.sequences, torch.Tensor(next_token.tolist())], dim=1)
            next_model_kwargs = self._update_model_kwargs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=torch.Tensor(next_token.tolist()),
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        while sample_search_cond_fn(state):
            state = sample_search_body_fn(state)

        last_lowcoder_states = self.module.transformer.cached_array

        encoded_lowcoder_states = self.preret_forward(
            hidden_states=last_lowcoder_states,
            attention_mask=torch.ones(last_lowcoder_states.shape[:-1]))

        return state, encoded_lowcoder_states



class RPTModel(RPTPreTrainedModel):
    module_class = RPTModule


class RPTForCausalLMModule(nn.Module):
    config: RPTConfig
    dtype: torch.dtype = torch.float32
    param_dtype: torch.dtype = torch.float32

    def _encode_forward(self, hidden_states, attention_mask, **kwargs):
        return self.transformer.retriever.preret_encode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

    def _lowcoder_forward(self, input_ids, attention_mask, **kwargs):
        """

        """
        lowcoder_outputs = self.transformer.lowcoder(
            self.transformer.wte(torch.Tensor(input_ids.astype("i4")).type(torch.int)),
            None,
            torch.Tensor(attention_mask),
            init_cache=True,
            **kwargs
        )

        outputs = self.transformer.retriever.preret_encode(
            hidden_states=torch.Tensor(lowcoder_outputs.last_hidden_state),
            attention_mask=torch.Tensor(attention_mask),
            **kwargs
        )

        return outputs, lowcoder_outputs.past_key_values

    def _augment_forward(self,
                         hidden_states,
                         neighbor_hidden_states,
                         neighbor_mask,
                         **kwargs):
        return self.transformer.upcoder.augment(
            hidden_states,
            neighbor_hidden_states,
            neighbor_mask,
            **kwargs
        )

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.transformer = RPTModule(self.config, dtype=self.dtype)
        # TODO: initialize
        self.lm_head = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.vocab_size,
            dtype=self.dtype,
            bias=False,
        )

    def forward(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
            deterministic: bool = True,
            retriever_supervision: RetrieverSupervision = None,
            train_step: Optional[int] = None,
            init_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            upcoder_input=None,
            encoded_neighbors: Optional[EncodedNeighbors] = None,

    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.broadcast_to(
                torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0),
                (batch_size, seq_length)
            )
        transformer_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            upcoder_input=upcoder_input,
            encoded_neighbors=encoded_neighbors,
            past_key_values=past_key_values,
        )
        if retriever_supervision is not None:
            transformer_input.update(retriever_supervision=retriever_supervision)

        def transformer(**kwargs):
            return self.transformer(
                **kwargs,
                deterministic=deterministic,
                train_step=train_step,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        outputs = transformer(**transformer_input)

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]

        lm_logits = self.unembed(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:] # TODO: Handle cache
        return FlaxRPTLMOutput(
            logits=lm_logits,
            upcoder_hidden_states=outputs.upcoder_hidden_states,
            upcoder_attentions=outputs.upcoder_attentions,
            cross_attentions=outputs.cross_attentions,
            lowcoder_last_hidden_state=outputs.lowcoder_last_hidden_state,
            lowcoder_hidden_states=outputs.lowcoder_hidden_states,
            lowcoder_attentions=outputs.lowcoder_attentions,
            retriever_output=outputs.retriever_output,
            retriever_input=outputs.retriever_input,
            past_key_values=outputs.past_key_values,
        )

    def unembed(self, hidden_states):
        if self.config.tie_word_embeddings:
            lm_logits = torch.Tensor(hidden_states[0] @ self.transformer.wte.weight.T).unsqueeze(0)
        else:
            lm_logits = self.lm_head(hidden_states)

        if self.config.palm_init:
            lm_logits = lm_logits / hidden_states.shape[-1]**0.5


        return lm_logits



class RPTForCausalLM(RPTPreTrainedModel):
    module_class = RPTForCausalLMModule
    def prepare_inputs_for_generation(self, input_ids,
                                      max_length=None,
                                      attention_mask=None,
                                      encoded_neighbors=None,
                                      past_key_values=None,
                                      **model_kwargs
                                      ):
        # TODO: Caching
        # initializing the cache
        #batch_size, seq_length = input_ids.shape
        #if past_key_values is None:
        #    past_key_values = self.init_cache(batch_size, self.config.window_length)

        return {
            "past_key_values": past_key_values,
            "encoded_neighbors": encoded_neighbors,
            "attention_mask": attention_mask,
        }

    def _update_model_kwargs_for_generation(self, model_outputs, model_kwargs, is_encoder_decoder: bool = False, standardize_cache_format: bool = False,):
        model_kwargs["past_key_values"] = model_outputs.past_key_values

        if model_kwargs.get("encoded_neighbors", None) is not None:
            encoded_neighbors = model_kwargs["encoded_neighbors"]
            if encoded_neighbors.chunk_index is None:
                chunk_index = torch.zeros([1, 1], dtype=torch.int32)  # this assumes bs=1
            else:
                chunk_index = torch.clip(encoded_neighbors.chunk_index + 1,
                                       max=self.config.chunk_size - 1)  # will need to modify later

            encoded_neighbors = EncodedNeighbors(
                neighbor_hidden_states=encoded_neighbors.neighbor_hidden_states[-1:, ...],  # assumes bs=1
                neighbor_mask=encoded_neighbors.neighbor_mask[-1:, ...],  # assumes bs=1
                chunk_index=chunk_index,
            )
            model_kwargs["encoded_neighbors"] = encoded_neighbors
        model_kwargs['attention_mask'] = torch.ones([1, 1], dtype=torch.int32)
        return model_kwargs