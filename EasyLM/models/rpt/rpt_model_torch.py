import json
from collections import namedtuple

import einops
import gin
import numpy as np
import optax
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from mlxu import function_args_to_config, load_pickle, open_file
from ml_collections import ConfigDict
from typing import Optional, Tuple, Union, Dict
from transformers import AutoTokenizer
from einops import rearrange
from torch_utils import make_attention_mask, make_causal_mask, combine_masks
from dataclasses import dataclass

import jax

# used just for the attention function call
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights
from EasyLM.memory_efficient_attention import dot_product_attention_multihead as efficient_dot_product_attention


RetrieverSupervision = namedtuple('RetrieverSupervision', ['nei_scores', 'nei_idx'])
EncodedNeighbors = namedtuple('EncodedNeighbors', ['neighbor_hidden_states', 'neighbor_mask',"chunk_index"])

@dataclass
class FlaxBaseModelOutput:
    last_hidden_state: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None

@dataclass
class FlaxBaseModelOutputCrossAttentions:
    last_hidden_state: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None


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
class FlaxRPTLMOutput:
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


def m1_cosine_decay_schedule(
    decay_steps: int,
    min_value:float,
    max_value:int,
    exponent: float = 1.0,
):
  if not decay_steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive decay_steps!')
  def schedule(count):
    count = jnp.minimum(count, decay_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
    decayed = 1-(cosine_decay ** exponent)
    decayed = (1 - min_value) * decayed + min_value
    return max_value*decayed

  return schedule

def topk_chunks(retriever_scores,num_candidates,*,where=None):
    # TODO: This used to have a @jax.vmap annotation on it, let's pytorch it
    def _topk_chunks(retriever_scores):
        return (-retriever_scores).argsort()[:num_candidates] #k = num_candidates
    if where is not None:
        retriever_scores = jnp.where(where,retriever_scores,-jnp.inf)
    return _topk_chunks(retriever_scores)

def create_segment_mask(total_num_chunks,n_skip_chunks):

    # TODO: This used to have a @jax.vmap annotation on it, let's pytorch it
    def _create_segment_mask(chunk_index):
        max_chunk = n_skip_chunks*(chunk_index//n_skip_chunks)
        return jnp.arange(total_num_chunks)<max_chunk - 2
    return _create_segment_mask(jnp.arange(total_num_chunks))


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


class TorchRPTRMSNorm(nn.Module):
    """
    RMS normalization layer
    """

    def __init__(self, config, dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.eps = self.config.rms_norm_eps
        if self.config.rms_one_baseline:
            self.weight = nn.Parameter(torch.zeros(self.config.hidden_size, dtype=self.param_dtype))
        else:
            self.weight = nn.Parameter(torch.ones(self.config.hidden_size, dtype=self.param_dtype))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.square(x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.type(torch.promote_types(self.dtype, torch.float32))
        output = self._norm(x).type(self.dtype)
        weight = torch.asarray(self.weight, dtype=self.dtype)
        if self.config.rms_one_baseline:
            return output * (1 - weight)
        else:
            return output * weight


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


class TorchRPTAttention(nn.Module):
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
        # E: positional encoding
        self.freqs_cis = precompute_freqs_cis(
            rot_dim,
            config.max_sequence_length * 2,
            dtype=self.dtype,
        )
        if self.config.add_null_attn:
            self.null_k = self.param(f'null_k', torch.nn.init.normal(0.0001),
                                     (1, 1, self.num_heads, self.head_dim))
            self.null_v = self.param(f'null_v', torch.nn.init.normal(0.0001),
                                     (1, 1, self.num_heads, self.head_dim))

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def _concatenate_to_cache(self, key, value, query, attention_mask):
        raise NotImplementedError("Not implemented yet")

    # TODO: Remember you removed to position_id parameter
    def forward(
            self,
            hidden_states,
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

        xq, xk, xv = self.wq.forward(hidden_states), self.wk.forward(hidden_states), self.wv.forward(hidden_states)

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)

        query_length = xq.shape[-3]
        batch_size = hidden_states.shape[0]
        query_attention_mask = attention_mask
        # TODO: Caching
        #if (self.has_variable("cache", "cached_key") or init_cache) and not disable_cache:
        #    xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)
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

        # TODO: Figure out caching
        #if self.has_variable("cache", "cached_key"):
        #    causal_mask = make_attention_mask(position_ids, position_ids_k, lambda x, y: x >= y,
        #                                      extra_batch_dims=0, dtype=bool)
        #else:
        #    causal_mask = self.causal_mask[:, :, :query_length, :key_length]
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
        attention_bias = torch.Tensor(np.select(
            attention_mask > 0,
            torch.full(attention_mask.shape, 0.0).type(self.dtype),
            torch.full(attention_mask.shape, torch.finfo(self.dtype).min).type(self.dtype),
        ))

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
        attn_output = self.wo.forward(attn_output)
        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs

    def concat_null_kv(self, xv, xk, attention_bias):

        attention_bias_shape = np.array(attention_bias.shape)
        attention_bias_shape[-1] = 1
        xk_shape = np.array(xk.shape)
        xk_shape[-3] = 1

        null_k = torch.broadcast_to(self.null_k, xk_shape)
        null_v = torch.broadcast_to(self.null_v, xk_shape)
        xk = torch.concatenate((xk, null_k), dim=-3)
        xv = torch.concatenate((xv, null_v), dim=-3)
        attention_bias = torch.concatenate((attention_bias, torch   .full(attention_bias_shape, 0.0)), dim=-1)
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


class TorchRPTCrossAttention(nn.Module):

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
        # TODO: Fishy. including the other initialization and null attention
        self.null_k = self.param(f'null_k', torch.nn.init.normal(0.0001), (1, 1, self.num_heads, self.head_dim))
        self.null_v = self.param(f'null_v', torch.nn.init.normal(0.0001), (1, 1, self.num_heads, self.head_dim))

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
    ) -> Tuple[torch.ndarray]:

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
            attention_mask = torch.Tensor([[attention_mask]])
            # TODO: Get rid of numpy
            attention_bias = np.select(
                attention_mask > 0,
                torch.full(attention_mask.shape, 0.0).type(self.dtype),
                torch.full(attention_mask.shape, torch.finfo(self.dtype).min).type(self.dtype),
            )
        else:
            attention_bias = None

        # TODO: Get rid of JAX
        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=jnp.float32,
            precision=self.precision,
        )
        attn_weights = torch.Tensor(attn_weights.tolist())

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, xv)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class TorchRPTMLP(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype

        # TODO: Don't forget to initialize
        self.w1 = nn.Linear(
            config.intermediate_size if config.gated_ff else 4 * config.hidden_size,
            dtype=self.dtype,
            bias=False,
        )
        self.w2 = nn.Linear(
            config.hidden_size,
            dtype=self.dtype,
            bias=False,
        )
        if self.config.gated_ff:
            self.w3 = nn.Linear(
                config.intermediate_size,
                dtype=self.dtype,
                bias=False,
            )
        # TODO: WHAT IS THIS??
        self.dropout = nn.Dropout(p=self.config.resid_pdrop, broadcast_dims=(0,))

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        if self.config.gated_ff:
            x1 = torch.silu(self.w1(x))
            x3 = self.w3(x)
            if self.config.mult_in_complex:
                x = mult_in_complex(x1, x3)
            else:
                x = x1 * x3
            x = self.w2(x)

            x = self.dropout(x, deterministic=deterministic)
        else:
            x = torch.gelu(self.w1(x))
            x = self.dropout(x, deterministic=deterministic)
            x = self.w2(x)

        return x


class TorchRPTLowcoderLayer(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        attention_module = TorchRPTAttention
        if self.config.remat_attention != '':
            attention_module = remat(
                TorchRPTAttention, static_argnums=(3, 4, 5, -1),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)
            )
        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
        )
        mlp_module = TorchRPTMLP
        if self.config.remat_mlp != '':
            mlp_module = remat(
                TorchRPTMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp)
            )

        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype
        )
        self.attention_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)

    def forward(
            self,
            hidden_states,
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
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
            sliding_window=self.config.sliding_window,
        )

        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

        # nomalize hidden states
        feed_forward_input = self.ffn_norm(hidden_states)

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
            feed_forward_hidden_states = self.feed_forward(
                feed_forward_input,
                deterministic,
            )

        # E: Add the goddamn linear layer output to the hidden states?
        hidden_states = hidden_states + feed_forward_hidden_states

        # what's on attn_output[1:]??
        return (hidden_states,) + attn_outputs[1:]


class TorchRPTLowcoderLayerCollection(nn.Module):
    """
    Basic series of masked attention encoders
    """

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        block = TorchRPTLowcoderLayer
        if self.config.remat_block != '':
            block = remat(
                TorchRPTLowcoderLayer, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        assert (self.config.num_hidden_layers % 2) == 0, f"config.num_hidden_layers should be devisible by 2"
        num_hidden_layers = self.config.num_hidden_layers // 2
        print("In Lowcoder: Using {} layers".format(num_hidden_layers))
        self.blocks = [
            block(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            ) for i in range(num_hidden_layers)
        ]

    def forward(
            self,
            hidden_states,
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

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids,
                deterministic,
                init_cache,
                output_attentions,
                fcm_mask,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return outputs

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class TorchRPTLowcoder(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.layers = TorchRPTLowcoderLayerCollection(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states,
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

        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class TorchRPTChunkedCrossAttention(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        self.chunk_size = self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.cross_attention = TorchRPTCrossAttention(self.config, dtype=self.dtype)

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
            hidden_states = torch.pad(hidden_states[:, causal_padding:, :], ((0, 0), (0, causal_padding), (0, 0)), 'constant')
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
            cross_attention_out = torch.pad(cross_attention_out, ((0, 0), (causal_padding, 0), (0, 0)), 'constant')[:,
                                  :-causal_padding]
        cross_attention_out = cross_attention_out.reshape([num_devices, seq_len, hidden_dim])
        return (cross_attention_out,) + output[1:]


class FlaxRPTUpcoderLayer(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, has_cca: bool = False):
        super(FlaxRPTUpcoderLayer, self)

        attention_module = TorchRPTAttention
        if self.config.remat_attention != '':
            attention_module = remat(
                TorchRPTAttention, static_argnums=(3, 4, 5, -1),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)
            )
        if self.has_cca:
            self.cca = TorchRPTChunkedCrossAttention(
                self.config,
                dtype=self.dtype
            )
            self.cca_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
            if self.config.use_cca_norm2:
                self.cca_norm2 = TorchRPTRMSNorm(self.config, dtype=self.dtype)
            else:
                self.cca_norm2 = None

        else:
            self.cca = None
            self.cca_norm = None

        self.attention = attention_module(
            self.config,
            dtype=self.dtype
        )
        mlp_module = TorchRPTMLP
        if self.config.remat_mlp != '':
            mlp_module = remat(
                TorchRPTMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp)
            )

        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype
        )
        self.attention_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        self.ffn_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)

    def forward(
            self,
            hidden_states,
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
            attention_mask,
            position_ids,
            deterministic,
            init_cache,
            output_attentions,
            fcm_mask,
            sliding_window=self.config.sliding_window,
        )
        attn_output = attn_outputs[0]
        hidden_states = hidden_states + attn_output

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
                deterministic,
            )

        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]


class FlaxRPTUpcoderLayerCollection(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dtype = dtype
        block = FlaxRPTUpcoderLayer
        if self.config.remat_block != '':
            block = remat(
                FlaxRPTUpcoderLayer, static_argnums=(6, 7, 8, 9),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        assert (self.config.num_hidden_layers % 2) == 0, f"config.num_hidden_layers should be divisible by 2"
        num_hidden_layers = self.config.num_hidden_layers // 2
        print("In Upcoder: Using {} layers".format(num_hidden_layers))

        def has_cca(layer_index):
            if self.config.cca_freq is None or self.config.cca_freq == 0:
                return False
            return (layer_index % self.config.cca_freq) == 0

        self.blocks = [
            block(
                self.config,
                dtype=self.dtype,
                has_cca=has_cca(i),
            ) for i in range(num_hidden_layers)
        ]

    def forward(
            self,
            hidden_states,
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

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,  # 0
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

            if output_attentions:
                all_attentions += (layer_outputs[1],)
                if block.has_cca:
                    all_cross_attentions += (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # this contains possible `None` values - `FlaxGPTJModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        if not return_dict:
            return outputs

        return FlaxBaseModelOutputCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlaxRPTNeighborAugmentor(nn.Module):

    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.postret_bidir_attention = TorchRPTCrossAttention(self.config, dtype=self.dtype)
        self.postret_bi_attention_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        self.query_nei_xattention_qnorm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        self.query_nei_xattention_knorm = TorchRPTRMSNorm(self.config, dtype=self.dtype)

        self.query_nei_xattention = TorchRPTCrossAttention(self.config, dtype=self.dtype)

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


class FlaxRPTCrossNeighborAugmentor(nn.Module):
    def __init__(self, config: RPTConfig, device_count, num_devices_chunks, num_neighbors, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.cross_neig_causal_att = TorchRPTAttention(self.config, dtype=self.dtype)
        self.xnei_norm1 = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        self.xnei_norm2 = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        # [device_count, num_devices_chunks*num_neighbors, 1]
        # TODO: handle initialization with dense init
        # TODO: missing in_features, need to run to figure out
        self.weight = nn.Linear(in_features=device_count*num_devices_chunks*num_neighbors, out_features=1, dtype=self.dtype, bias=True)
        if self.config.use_xnei_bias:
            self.xnei_bias = self.param(f'xnei_bias', jax.nn.initializers.normal(0.01),
                                        (1, 1, 1, self.config.hidden_size))
        else:
            self.xnei_bias = None

    def forward(self,
                 neighbor_hidden_states: torch.Tensor,
                 neighbor_mask: torch.Tensor,
                 output_attentions: torch.Tensor,
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
            attention_mask=pooled_neighbor_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
            sliding_window=False,
            # disable_cache=False,
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
        return (neighbor_hidden_states,) + cross_neig_out[1:]


class FlaxRPTUpcoder(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype

        if self.config.augment_neighbors:
            self.neighbor_augmentor = FlaxRPTNeighborAugmentor(self.config, dtype=self.dtype)
        else:
            self.neighbor_augmentor = None

        if self.config.augment_across_neighbors:
            # TODO: Fix parameters
            self.neighbor_cross_augmentor = FlaxRPTCrossNeighborAugmentor(self.config, dtype=self.dtype)
        else:
            self.neighbor_cross_augmentor = None
        self.layers = FlaxRPTUpcoderLayerCollection(self.config, dtype=self.dtype)

    def augment(self,
                hidden_states: torch.Tensor,
                neighbor_hidden_states: torch.Tensor,
                neighbor_mask: torch.Tensor,
                deterministic: bool = True,
                output_attentions: bool = False,
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
        if self.neighbor_cross_augmentor is not None and neighbor_hidden_states is not None:
            nei_xaug_outputs = self.neighbor_cross_augmentor(
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
                device_count=hidden_states.shape[0],
            )
            neighbor_hidden_states = nei_xaug_outputs[0]
        return neighbor_hidden_states

    def forward(
            self,
            hidden_states,
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
        if chunk_index is None:
            neighbor_hidden_states = self.augment(
                hidden_states,
                neighbor_hidden_states,
                neighbor_mask,
                deterministic,
                output_attentions,
            )
        # else We are generating... And have already augmented the neighbor hidden states.

        outputs = self.layers(
            hidden_states,
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

        if not return_dict:
            return outputs

        return FlaxBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxRPTRetriever(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.preret_bidir_attention = TorchRPTCrossAttention(self.config, dtype=self.dtype)
        self.preret_bi_attention_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        self.pre_key_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        # TODO: handle initialization
        # TODO: handle input size
        self.key_projection = nn.Linear(
            out_features=self.config.hidden_size,
            dtype=self.dtype,
            bias=True
        )
        self.pre_query_norm = TorchRPTRMSNorm(self.config, dtype=self.dtype)
        # TODO: handle initialization
        # TODO: handle input size
        self.query_projection = nn.Linear(
            out_features=self.config.hidden_size,
            dtype=self.dtype,
            bias=True,
        )
        self.fill_value = self.config.retriever_fill_value
        self.n_skip_chunks = (self.config.max_sequence_length // self.config.n_windows) // self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.threshold_nei_scores = self.config.threshold_nei_scores
        self.num_sequence_chunks = self.config.num_sequence_chunks
        if self.config.aux_loss_schedule_steps is not None:
            assert self.config.aux_scale is not None
            self.aux_scale = self.config.aux_scale
            self.aux_loss_schedule_fn = optax.linear_schedule(0, 1, self.config.aux_loss_schedule_steps)

        if self.config.max_margin is not None and self.config.margin_schedule_steps is not None:
            assert self.config.max_margin >= 1
            self.increase_margin_schedule_fn = optax.linear_schedule(1, self.config.max_margin,
                                                                     self.config.margin_schedule_steps)

        if self.config.ss_schedule_steps is not None and \
                self.config.scheduled_sampling_max_prob is not None \
                and self.config.scheduled_sampling_min_prob is not None \
                and self.has_rng("dropout"):
            self.ss_rng = self.make_rng("dropout")
            self.scheduled_sampling_schedule_fn = m1_cosine_decay_schedule(decay_steps=self.config.ss_schedule_steps,
                                                                           min_value=self.config.scheduled_sampling_min_prob,
                                                                           max_value=self.config.scheduled_sampling_max_prob)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            retriever_supervision: RetrieverSupervision = None,
            train_step: Optional[int] = None,
            deterministic: bool = True,
            output_attentions: bool = False,
    ):

        encoded_output = self.preret_encode(
            hidden_states,
            attention_mask,
            deterministic,
            output_attentions)

        query_based_scores = torch.einsum('qd,kd->qk', encoded_output.query_chunks,
                                        encoded_output.key_chunks)
        query_based_scores /= torch.sqrt(self.config.hidden_size)

        segment_mask = create_segment_mask(query_based_scores.shape[0], self.n_skip_chunks)

        chunk_mask = encoded_output.chunk_mask
        chunk_mask &= segment_mask
        # TODO: Might not be relevant in inference
        if retriever_supervision is not None:
            aux_loss, target_neighbor_mask, target_score_based_idx, ret_metrics = self.compute_retriever_loss(
                query_based_scores,
                retriever_supervision,
                chunk_mask,
                train_step)
        else:
            aux_loss = None
            target_neighbor_mask = None
            target_score_based_idx = None
            ret_metrics = None

        query_score_based_idx = topk_chunks(query_based_scores, num_candidates=self.num_neighbors, where=chunk_mask)

        top_nei_idx, nei_mask = self.apply_scheduled_sampling(
            query_score_based_idx,
            torch.take_along_axis(chunk_mask, query_score_based_idx, axis=-1),
            target_score_based_idx,
            target_neighbor_mask,
            train_step,
            deterministic)
        neighbor_hidden_states = self.lookup_neighbor_states(encoded_output.encoded_hidden_states, top_nei_idx)
        # TODO: fishy unsqueeze
        nei_mask = torch.broadcast_to(nei_mask.unsqueeze(), neighbor_hidden_states.shape[:-1])
        return FlaxRPTRetrieverNeighborOutput(aux_loss=aux_loss if aux_loss is not None else None,
                                              loss_scale=self.get_loss_scale(
                                                  train_step) if aux_loss is not None else None,
                                              neighbor_hidden_states=neighbor_hidden_states,
                                              neighbor_mask=nei_mask,
                                              retrieval_metrics=jax.tree_map(lambda x: x.mean(),
                                                                             ret_metrics) if ret_metrics is not None else None,
                                              )

    # @classmethod
    def lookup_neighbor_states(cls, cand_hidden_states: torch.Tensor, top_nei_idx: torch.Tensor):
        num_document_chunks = top_nei_idx.shape[0]
        shifted_hidden_states = torch.pad(cand_hidden_states[1:, ...], ((0, 1), (0, 0), (0, 0)))
        curr_neighbor_hidden_states = cand_hidden_states[top_nei_idx.reshape(-1)]
        next_neighbor_hidden_states = shifted_hidden_states[top_nei_idx.reshape(-1)]
        neighbor_hidden_states = torch.concatenate((curr_neighbor_hidden_states, next_neighbor_hidden_states), dim=-2)
        neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, '(b k) r d -> b k r d', b=num_document_chunks)
        return neighbor_hidden_states

    def preret_encode(self,
                      hidden_states: torch.Tensor,
                      attention_mask: torch.Tensor,
                      deterministic: bool,
                      output_attentions: bool = False, ):
        original_hidden_states_shape = hidden_states.shape
        original_attention_mask_shape = attention_mask.shape

        # TODO: verify equivilance
        original_hidden_states = einops.rearrange(hidden_states, 'b l (c d) -> (b l) c d', c=self.config.chunk_size)

        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))

        attention_mask = einops.rearrange(attention_mask, 'b l d -> (b l) d')

        # add a chunk dimension
        # 1. apply bi-dir attention
        preret_bi_output = self.preret_bidir_attention.forward(
            self.preret_bi_attention_norm(original_hidden_states),
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions)
        encoded_hidden_states = preret_bi_output[0] + original_hidden_states

        # 2. pool
        pooled_hidden_states = encoded_hidden_states.mean(dim=-2)

        # 3. project to query chunks and key chunks
        key_chunks = self.key_projection.forward(self.pre_key_norm.forward(pooled_hidden_states))
        query_chunks = self.query_projection.forward(self.pre_query_norm.forward(pooled_hidden_states))
        chunk_mask = attention_mask.type(bool).any(-1)[..., None]
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

    def apply_scheduled_sampling(self,
                                 query_score_based_idx: torch.Tensor,
                                 chunk_mask: torch.Tensor,
                                 target_score_based_idx: torch.Tensor,
                                 target_neighbor_mask: torch.Tensor,
                                 train_step,
                                 deterministic):
        if deterministic or self.is_initializing() or target_score_based_idx is None:
            top_nei_idx, top_nei_mask = query_score_based_idx, chunk_mask
        else:

            # TODO: figure it out, might not be relevant due to inference only code
            rv = torch.bernoulli(p=self.scheduled_sampling_schedule_fn(train_step if not self.is_initializing() else 1),
                                      shape=())  # this is a boolean of shape [1]
            top_nei_idx, top_nei_mask = jax.lax.cond(rv,
                                                     (), lambda args: (query_score_based_idx, chunk_mask),
                                                     (), lambda args: (target_score_based_idx, target_neighbor_mask)
                                                     )
        return top_nei_idx, top_nei_mask

    def get_loss_scale(self, train_step):
        loss_scale = self.aux_loss_schedule_fn(train_step if not self.is_initializing() else 1)
        return loss_scale * self.aux_scale
# loss is calculated as lm_loss + (raw_aux_loss/valid_pairs.sum())* self.get_loss_scale(train_step)

class FlaxRPTModule(nn.Module):
    def __init__(self, config: RPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        # the embedding dim
        self.embed_dim = self.config.hidden_size

        # TODO: move this wte and dropout into the lowcoder.

        # define a dropout layer
        self.dropout = nn.Dropout(p=self.config.embd_pdrop)

        # TODO: handle init
        self.wte = nn.Embedding(
            self.config.vocab_size,  # input size
            self.config.hidden_size,  # embedding size
            dtype=self.dtype
        )

        """
        # word to embedding module (layer)
        self.wte = nn.Embed(
            self.config.vocab_size,  # input size
            self.config.hidden_size,  # embedding size
            embedding_init=dense_init(self.config, is_embedding=True),
            # basically np.random of weights in the correct size
            dtype=self.dtype,  # type of embedding vector entries
            param_dtype=self.param_dtype,  # type of input
        )
        """

        self.lowcoder = TorchRPTLowcoder(self.config, dtype=self.dtype)
        if self.config.cca_freq is not None and self.config.cca_freq > 0:
            self.retriever = FlaxRPTRetriever(self.config, dtype=self.dtype)
        else:
            self.retriever = None

        self.upcoder = FlaxRPTUpcoder(self.config, dtype=self.dtype)

        # TODO: move this ln_f into the upcoder.
        self.ln_f = TorchRPTRMSNorm(self.config, dtype=self.dtype)

    # TODO: Handle
    def _concatenate_to_lowcoder_cache(self, array):
        chunk_size = self.config.chunk_size
        is_initialized = self.has_variable("cache", "cached_array")
        *batch_dims, _, hidden_dim = array.shape
        cached_array = self.variable("cache", "cached_array",
                                     jnp.zeros,
                                     tuple(batch_dims) + (self.config.chunk_size, hidden_dim),
                                     array.dtype)
        if is_initialized:
            last_chunk = array[..., -chunk_size:, :]

            num_updated_cache_vectors = last_chunk.shape[-2]
            shift = self.config.chunk_size - num_updated_cache_vectors  # will need to update if I change retrieval stride
            indices = (0,) * len(batch_dims) + (shift, 0)

            array_operand = torch.roll(cached_array.value, shift=-num_updated_cache_vectors, axis=-2)
            cached_array.value = lax.dynamic_update_slice(array_operand,
                                                          last_chunk,
                                                          indices)

    def forward(
            self,
            input_ids,
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
            input_embeds = self.wte.forward(input_ids.astype("i4"))

            # TODO: Determinsitc
            hidden_states = self.dropout.forward(input_embeds)

            lowcoder_outputs = self.lowcoder.forward(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = lowcoder_outputs.last_hidden_state if return_dict else lowcoder_outputs[0]
            if self.has_variable("cache", "cached_array") or init_cache:
                self._concatenate_to_lowcoder_cache(hidden_states)

            retriever_input = hidden_states
            if self.retriever is not None:
                if encoded_neighbors is not None:
                    neighbor_hidden_states = encoded_neighbors.neighbor_hidden_states
                    neighbor_mask = encoded_neighbors.neighbor_mask
                    chunk_index = encoded_neighbors.chunk_index
                else:
                    retriever_output = self.retriever.forward(hidden_states=retriever_input,
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

        upcoder_outputs = self.upcoder.forward(
            hidden_states,
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

        hidden_states = upcoder_outputs.last_hidden_state if return_dict else upcoder_outputs[0]
        hidden_states = self.ln_f.forward(hidden_states)

        if not return_dict:
            return (hidden_states,) + upcoder_outputs + lowcoder_outputs

        return FlaxRPTModelOutput(
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
