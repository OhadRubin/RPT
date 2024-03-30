import json

import einops
import gin
import numpy as np
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from mlxu import function_args_to_config, load_pickle, open_file
from ml_collections import ConfigDict
from typing import Optional, Tuple, Union
from transformers import AutoTokenizer
from einops import rearrange
from torch_utils import make_attention_mask, make_causal_mask, combine_masks

import jax

# used just for the attention function call
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights
from EasyLM.memory_efficient_attention import dot_product_attention_multihead as efficient_dot_product_attention



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

    def __init__(self, config, dtype, param_dtype):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
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

    def __init__(self, config: RPTConfig, dtype: torch.float32, param_dtype: torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # TODO: Use the initialization function
        self.wq = torch.nn.Parameter(
            torch.ones(
                self.embed_dim,
                config.num_attention_heads * self.head_dim,
                dtype=self.dtype)
        )

        self.wk = torch.nn.Parameter(
            torch.rand(
                self.embed_dim,
                config.num_attention_heads * self.head_dim,
                dtype=self.dtype)
        )
        self.wv = torch.nn.Parameter(
            torch.rand(
                self.embed_dim,
                config.num_key_value_heads * self.head_dim,
                dtype=self.dtype)
        )

        self.wo = torch.nn.Parameter(
            torch.rand(
                self.embed_dim,
                config.hidden_size, dtype=self.dtype)
        )

        # self.resid_dropout = nn.Dropout(rate=config.resid_pdrop,broadcast_dims=(0,))

        # TODO: Bruh mask (not actually masking anything)
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

        hidden_states_for_mult = torch.permute(hidden_states, (0, 2, 1))

        # wq projects the input embeddings into the query embeddings
        # query embeddings = d_q, input_embeddings = d
        # Wq = (d_q x d)
        # input_embeddings = (len, d)
        # input_embeddings * Wq.T = (Wq x input_embeddings.T).T

        # TODO: let's figure this out
        xq, xk, xv = self.wq.matmul(hidden_states_for_mult).permute((0, 2, 1)), self.wk.matmul(hidden_states_for_mult).permute((0, 2, 1)), self.wv.matmul(hidden_states_for_mult).permute((0, 2, 1))

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
        attn_output = self.wo.matmul(attn_output.permute((0, 2, 1))).permute((0, 2, 1))
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


def dense_init(input_tensor, config, is_embedding=False):
    if config.palm_init:
        if is_embedding:
            return torch.nn.init.normal(tensor=input_tensor, std=1.0)
        # TODO: The use of len needs more examination
        return torch.nn.init.normal(tensor=input_tensor, std=torch.sqrt(config.initializer_range / len(input_tensor)))
    else:
        return torch.nn.init.normal(tensor=input_tensor, std=config.initializer_range)
