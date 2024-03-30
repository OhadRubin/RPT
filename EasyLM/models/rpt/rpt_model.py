from typing import Dict, Tuple, Union
import json


import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
import einops

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from ml_collections import ConfigDict
from mlxu import function_args_to_config, load_pickle, open_file

from EasyLM.memory_efficient_attention import dot_product_attention_multihead as efficient_dot_product_attention
from EasyLM.jax_utils import (
    get_jax_mesh, get_gradient_checkpoint_policy, create_target_scores, add_process_dim, remove_process_dim
)
from einops import rearrange
import flax
import gin

from collections import namedtuple
import optax
import rax
import operator
from typing import Optional
from transformers.utils import ModelOutput
from transformers import AutoTokenizer
RetrieverSupervision = namedtuple('RetrieverSupervision', ['nei_scores', 'nei_idx'])
# EncodedNeighbors = namedtuple('EncodedNeighbors', ['neighbor_hidden_states', 'neighbor_mask',"retriever_input"])    
EncodedNeighbors = namedtuple('EncodedNeighbors', ['neighbor_hidden_states', 'neighbor_mask',"chunk_index"])    
# from flax.struct import PyTreeNode

# @flax.struct.dataclass
# class EncodedNeighbors(PyTreeNode):
#     neighbor_hidden_states: jnp.ndarray = None
#     neighbor_mask: jnp.ndarray = None
#     retriever_input: Optional[jnp.ndarray] = None

def dense_init(config,is_embedding=False):
    if config.palm_init:
        if is_embedding:
            return jax.nn.initializers.normal(stddev=1.0)
        return jax.nn.initializers.variance_scaling(config.initializer_range, 'fan_in', 'normal', out_axis=0)
    else:
        return jax.nn.initializers.normal(stddev=config.initializer_range)

@flax.struct.dataclass
class FlaxBaseModelOutputCrossAttentions(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    

@flax.struct.dataclass
class FlaxRPTRetrieverEncodedOutput(ModelOutput):
    original_hidden_states: jnp.ndarray = None
    encoded_hidden_states: jnp.ndarray = None
    attention_mask: jnp.ndarray = None
    key_chunks: jnp.ndarray = None
    query_chunks: jnp.ndarray = None
    chunk_mask: jnp.ndarray = None
    preret_attention: Optional[jnp.ndarray] = None
    position_ids: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxRPTRetrieverEncodedOutput(ModelOutput):
    original_hidden_states: jnp.ndarray = None
    encoded_hidden_states: jnp.ndarray = None
    attention_mask: jnp.ndarray = None
    key_chunks: jnp.ndarray = None
    query_chunks: jnp.ndarray = None
    chunk_mask: jnp.ndarray = None
    preret_attention: Optional[jnp.ndarray] = None
    position_ids: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxRPTRetrieverNeighborOutput(ModelOutput):
    aux_loss: jnp.ndarray = None
    loss_scale: jnp.ndarray = None
    neighbor_hidden_states: jnp.ndarray = None
    neighbor_mask: jnp.ndarray = None
    retrieval_metrics: Optional[Dict[str, jnp.ndarray]] = None
    
@flax.struct.dataclass
class FlaxRPTLowcoderRetrieverEncodedOutput(ModelOutput):
    hidden_states: jnp.ndarray = None
    attention_mask: jnp.ndarray = None
    neighbor_hidden_states: jnp.ndarray = None
    neighbor_mask: jnp.ndarray = None

@flax.struct.dataclass
class FlaxRPTModelOutput(ModelOutput):
    last_hidden_state: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    upcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    upcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_last_hidden_state: Optional[jnp.ndarray] = None
    lowcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    retriever_output: FlaxRPTRetrieverNeighborOutput = None
    retriever_input: Optional[jnp.ndarray] = None
    
@flax.struct.dataclass
class FlaxRPTLMOutput(ModelOutput):
    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
    upcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    upcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    cross_attentions: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_last_hidden_state: Optional[jnp.ndarray] = None
    lowcoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
    lowcoder_attentions: Optional[Tuple[jnp.ndarray]] = None
    retriever_output: FlaxRPTRetrieverNeighborOutput = None
    retriever_input: Optional[jnp.ndarray] = None


def m1_cosine_decay_schedule(
    decay_steps: int,
    min_value:float,
    max_value:int,
    exponent: float = 1.0,
):
  if not decay_steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive decay_steps!')
  @jax.profiler.annotate_function
  def schedule(count):
    count = jnp.minimum(count, decay_steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
    decayed = 1-(cosine_decay ** exponent)
    decayed = (1 - min_value) * decayed + min_value
    return max_value*decayed

  return schedule

@jax.profiler.annotate_function
def topk_chunks(retriever_scores,num_candidates,*,where=None):
    @jax.vmap
    def _topk_chunks(retriever_scores):
        return (-retriever_scores).argsort()[:num_candidates] #k = num_candidates
    if where is not None:
        retriever_scores = jnp.where(where,retriever_scores,-jnp.inf)
    return _topk_chunks(retriever_scores)

@jax.profiler.annotate_function
def create_segment_mask(total_num_chunks,n_skip_chunks):
    @jax.vmap
    def _create_segment_mask(chunk_index):
        max_chunk = n_skip_chunks*(chunk_index//n_skip_chunks)
        return jnp.arange(total_num_chunks)<max_chunk - 2 
    return _create_segment_mask(jnp.arange(total_num_chunks))

@jax.profiler.annotate_function
def compute_pairs(a, op):
  """Computes pairs based on values of `a` and the given pairwise `op`.

  Args:
    a: The array used to form pairs. The last axis is used to form pairs.
    op: The binary op to map a pair of values to a single value.

  Returns:
    A :class:`jax.Array` with the same leading dimensions as `a`, but with the
    last dimension expanded so it includes all pairs `op(a[..., i], a[..., j])`
  """
  a_i = jnp.expand_dims(a, -1)
  a_j = jnp.expand_dims(a, -2)
  result_shape = jnp.broadcast_shapes(a_i.shape, a_j.shape)
  result = jnp.broadcast_to(op(a_i, a_j), result_shape)
  out_shape = tuple(result.shape[:-2]) + (result.shape[-2] * result.shape[-1],)
  return jnp.reshape(result, out_shape)

@jax.profiler.annotate_function
def compute_retrieval_metrics(scores_pred, scores,query_mask,target_mask):
    scores_pred = jnp.where(query_mask, scores_pred,-jnp.inf)
    scores = jnp.where(target_mask, scores, 0.0)
    def recall(n):
        res = rax.recall_metric(scores=scores_pred,
                labels=(scores>0).astype(np.float32),
                topn=n,
                reduce_fn=None,
                where= query_mask)
        res_mask = combine_masks(
                        jnp.isfinite(res),
                        jnp.any(query_mask,axis=-1))
        return res.mean(where=res_mask)
    return {f"recall@{i}":recall(i) for i in [2,5,10,20]}
@jax.profiler.annotate_function
@jax.vmap
def compute_ndcg_lambda(scores_pred, scores,query_mask,target_mask):
    """
    Compute the NDCG delta matrix for given scores and predicted scores.
    
    Args:
        score_pred (numpy.array): predicted scores
        score (numpy.array): true scores
        
    Returns:
        numpy.array: NDCG delta matrix
    """
    # Get the descending order of indices based on scores
    scores_pred = jnp.where(query_mask, scores_pred,-jnp.inf)
    scores = jnp.where(target_mask, scores, 0.0)
    def recall(n):
        return rax.recall_metric(scores=scores_pred,
                labels=scores,
                topn=n,
                where= target_mask,reduce_fn=None)
    metrics = {f"recall@{i}":recall(i) for i in [2,5,10,20,20]}

    argsort_score = jnp.argsort(scores)[::-1]
    argsort_score_pred = jnp.argsort(scores_pred)[::-1]

    # Calculate rank plus one for true scores and predicted scores
    rank_plus_one = jnp.argsort(argsort_score) + 2
    rank_plus_one_pred = jnp.argsort(argsort_score_pred) + 2

    # Calculate the numerator, which is the same for both IDCG and DCG
    numerator = 2 ** scores - 1

    # Calculate the denominators for IDCG and DCG
    idcg_denominator = jnp.log2(rank_plus_one)
    dcg_denominator = jnp.log2(rank_plus_one_pred)

    # Calculate IDCG and DCG
    idcg = numerator / idcg_denominator

    # Calculate the difference between numerators
    numerator_ij = numerator[:, None] - numerator

    # Calculate the difference between DCG denominators
    dcg_denominator_ij = dcg_denominator[:, None] - dcg_denominator
    # Calculate the NDCG delta matrix
    ndcg_delta_ij = jnp.abs((numerator_ij * dcg_denominator_ij) / jnp.maximum(jnp.sum(idcg), 0.001)).reshape(-1)
    return jnp.where(jnp.isfinite(ndcg_delta_ij), ndcg_delta_ij, 0.0)

@jax.profiler.annotate_function
def make_cross_mask(q_len, kv_len, extra_batch_dims: int = 0, dtype=bool):
    source_idxs = jnp.arange(kv_len-q_len, kv_len, dtype=jnp.int32)
    target_idxs = jnp.arange(kv_len, dtype=jnp.int32)
    return nn.make_attention_mask(source_idxs,target_idxs, lambda x,y:x>=y,
                                  extra_batch_dims=extra_batch_dims, dtype=dtype)

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
    >>> from transformers import RPTModel, RPTConfig
    >>> # Initializing a RPT rpt-7b style configuration
    >>> configuration = RPTConfig()
    >>> # Initializing a model from the rpt-7b style configuration
    >>> model = RPTModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
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
        sliding_window:bool=True,
        n_windows:int=1,
        window_length: int = 2048,
        cca_freq:Optional[int]=None,
        num_neighbors:Optional[int]=None,
        num_sequence_chunks:Optional[int]=None,
        chunk_size:Optional[int]=64,
        num_scored_neighbors:Optional[int]=None,
        mesh_dim:Optional[str]=None,
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
        return_ret_metrics:bool = True,
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
        self.window_length  = window_length
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
            self.num_sequence_chunks = max_sequence_length//chunk_size
        else:
            self.num_sequence_chunks = None
        self.num_document_chunks = self.document_length//chunk_size
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
    def get_jax_mesh(self, mesh_dim):
        return get_jax_mesh(mesh_dim, ('dp', 'fsdp', 'mp'))
    @classmethod
    def get_tokenizer(cls,**kwargs):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
                            pad_token='<|endoftext|>',
                            mask_token='<|endoftext|>',
                            **kwargs)
        return tokenizer

    @staticmethod
    def get_partition_rules():
        """ Parition rules for GPTJ. Note that these rules are orderd, so that
            the beginning rules match first. It is important to use
            PartitionSpec() instead of None here because JAX does not treat
            None as a pytree leaf.
        """
        return (
            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # atention
            ("(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            ("query_projection/kernel",PS("fsdp", "mp")),
            ("key_projection/kernel",PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            ('.*', PS(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple()

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')


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



remat = nn_partitioning.remat


class FlaxRPTRMSNorm(nn.Module):
    """
    RMS normalization layer
    """
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.eps = self.config.rms_norm_eps
        if self.config.rms_one_baseline:
            self.weight = self.param(
                'kernel',
                nn.initializers.zeros,
                (self.config.hidden_size,),
                self.param_dtype,
            )
        else:
            self.weight = self.param(
                'kernel',
                nn.initializers.ones,
                (self.config.hidden_size,),
                self.param_dtype,
            )


    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.eps)
    
    @jax.profiler.annotate_function
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = jnp.asarray(self.weight, self.dtype)
        if self.config.rms_one_baseline:
            return output * (1 - weight)
        else:
            return output * weight

@jax.profiler.annotate_function
def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = np.arange(end)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

@jax.profiler.annotate_function
def apply_rotary_emb_(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
    freqs_cis_k: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])

    # add head dim
    # freqs_cis = (1, 1024, 64)
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))

    # freqs_cis = (1, 1024, 1, 64)
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    if freqs_cis_k is None:
        xk_out = xk_ * freqs_cis
        xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    else:
        freqs_cis_k = jnp.reshape(freqs_cis_k, (*freqs_cis_k.shape[:2], 1, *freqs_cis_k.shape[2:]))
        xk_out = xk_ * freqs_cis_k
        xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)


@jax.profiler.annotate_function
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
    out = jnp.stack((real_part, imag_part), axis=-1).reshape(*real_part.shape[:-1], -1)
    
    return out

@jax.profiler.annotate_function
def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    dtype: jnp.dtype=jnp.float32,
    freqs_cis_k: jnp.ndarray = None,
    rot_dim: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if rot_dim is not None and rot_dim>0:

        # Separate the tensors based on the rotation dimensions
        xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
        xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

        # freqs_q_rot = freqs_q[..., :rot_dim]
        # freqs_k_rot = freqs_k[..., :rot_dim] if freqs_k is not None else None

        # Apply the function on the parts that need rotation
        print(freqs_cis.shape, xq_rot.shape, xk_rot.shape)
        xq_rot, xk_rot = apply_rotary_emb_(xq_rot, xk_rot, freqs_cis, dtype=dtype, freqs_cis_k=freqs_cis_k)

        # Concatenate the rotated and non-rotated parts
        xq_out = jnp.concatenate((xq_rot, xq_pass), axis=-1)
        xk_out = jnp.concatenate((xk_rot, xk_pass), axis=-1)
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
    hidden_states = jnp.broadcast_to(hidden_states,
                                     (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class FlaxRPTAttention(nn.Module):
    """
    The transformer's masked self attention layer
    """
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=my_dense_init(self.config),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            # config.num_attention_heads*self.head_dim,
            config.num_key_value_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=my_dense_init(self.config),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            # config.num_attention_heads*self.head_dim,
            config.num_key_value_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=my_dense_init(self.config),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=my_dense_init(self.config),
            precision=self.precision,
        )

        # self.resid_dropout = nn.Dropout(rate=config.resid_pdrop,broadcast_dims=(0,))

        # TODO: Bruh mask (not actually masking anything)
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_sequence_length), dtype="bool"), dtype="bool")
        if self.config.rot_dim is not None and self.config.rot_dim>0:
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
            self.null_k =  self.param(f'null_k', jax.nn.initializers.normal(0.0001), (1,1,self.num_heads,self.head_dim))
            self.null_v =  self.param(f'null_v', jax.nn.initializers.normal(0.0001), (1,1,self.num_heads,self.head_dim))

    @jax.profiler.annotate_function
    def _split_heads(self, hidden_states):
        """
        Split the hidden states (1, 1024, 4096) = (1, input_length, embedding_length)
        into 32 heads with 128 dims each. i.e each token embedding is broken into 32 token embeddings for 32 heads.
        [
            [ token 0 embedding "block", token 1 embedding "block", ..., token 1023 embedding "block"]
        ]
        """
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
    @jax.profiler.annotate_function
    def _merge_heads(self, hidden_states):
        """
        Merge the 32 embeddings back into a singular embeddings (i.e concatenation)
        """
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
    

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_mask = self.variable("cache", "cache_mask", jnp.zeros, attention_mask.shape, jnp.int32)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        attention_mask = attention_mask.astype(int)

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            num_updated_cache_vectors = query.shape[-3]
            shift = max_length-num_updated_cache_vectors
            
            cache_index.value = cache_index.value + num_updated_cache_vectors
            
            
            def cur_index_small(key, value, attention_mask):
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                attention_mask = jnp.arange(max_length) < cur_index + num_updated_cache_vectors
                attention_mask = attention_mask[None, :].astype(int)
                
                return key, value, attention_mask
            
            def cur_index_big(key, value, attention_mask):
                indices = (0,) * len(batch_dims) + (shift, 0, 0)

                key_operand = jnp.roll(cached_key.value, shift=-num_updated_cache_vectors, axis=-3)
                key = lax.dynamic_update_slice(key_operand,
                                            key,
                                            indices)

                value_operand = jnp.roll(cached_value.value, shift=-num_updated_cache_vectors, axis=-3)
                value = lax.dynamic_update_slice(value_operand,
                                                value,
                                                indices)

                mask_operand = jnp.roll(cache_mask.value, shift=-num_updated_cache_vectors, axis=-1)
                attention_mask = lax.dynamic_update_slice(mask_operand,
                                                        attention_mask.astype(mask_operand.dtype),
                                                        (0,) * len(batch_dims)+ (shift,))
                return key, value, attention_mask
            

            # cond_input = (key, value, attention_mask,)
            # key, value, attention_mask = lax.cond(cur_index < max_length,
            #                                     cur_index_small,
            #                                     cur_index_big,
            #                                     *cond_input)
            # else:
            key, value, attention_mask = cur_index_big(key, value, attention_mask)
                
            cached_key.value = key
            cached_value.value = value
            cache_mask.value = attention_mask

        return key, value, attention_mask.astype(jnp.int32)
    @jax.profiler.annotate_function
    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask=None,
        sliding_window=False,
        disable_cache:bool=False
    ):
        n_windows=self.config.n_windows
        # stride = self.config.stride if not disable_cache else None

        # (1x1024,4096) = (??,input size,embedding_dim)
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)
        
        
        query_length = xq.shape[-3]
        batch_size = hidden_states.shape[0]
        query_attention_mask = attention_mask
        if (self.has_variable("cache", "cached_key") or init_cache) and not disable_cache:
            xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)            
        key_length = xk.shape[-3]


            
        position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(query_attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, query_length)
            ).astype(int)
            
        with jax.profiler.TraceAnnotation("rotary_emb"):
            if key_length!=query_length:
                position_ids_k = jnp.broadcast_to(
                        jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                        (batch_size, key_length)
                    ).astype(int)
                freqs_cis_k = jnp.take(self.freqs_cis, position_ids_k, axis=0)
                position_ids += position_ids_k.max()-position_ids.max()
                freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
            else:
                position_ids_k = position_ids
                freqs_cis_k = None
            
        if sliding_window and n_windows>1:
            query_length = query_length//n_windows
            key_length = key_length//n_windows
            batch_size = batch_size*n_windows
            attention_mask = rearrange(attention_mask, 'b (s l) -> (b s) l',s=n_windows)
        

            
        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
            

        if self.has_variable("cache", "cached_key"): 
            causal_mask =  nn.make_attention_mask(position_ids, position_ids_k, lambda x,y:x>=y,
                                extra_batch_dims=0, dtype=bool)
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]
        with jax.profiler.TraceAnnotation("attention_mask"):        
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask, fcm_mask)

        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        with jax.profiler.TraceAnnotation("apply_rotary_emb__"):
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis,
                                    freqs_cis_k=freqs_cis_k,
                                    dtype=self.dtype, rot_dim=self.config.rot_dim)

        # transform boolean mask into float mask
        with jax.profiler.TraceAnnotation("attention_bias"):
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

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
            attn_output = efficient_dot_product_attention(
                xq,
                xk,
                xv,
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
            if sliding_window and n_windows>1:
                xq, xk, xv = map(lambda t: rearrange(t, 'b (s l) ... -> b s l ...',s=n_windows),
                                (xq, xk, xv))
                attention_bias = rearrange(attention_bias, '(b s) ... -> b s ...',s=n_windows)
                
                past_attention_bias = jnp.full(attention_bias.shape, 0.0)
                
                past_xk, past_xv, past_attention_bias = map(lambda t: t[:,:-1,:,:,:],
                                                            (xk, xv, attention_bias))
                past_xk, past_xv = map(lambda t: jnp.pad(t,[(0,0),(1,0),(0,0),(0,0),(0,0)]),
                                                            (past_xk, past_xv))
                past_attention_bias = jnp.pad(past_attention_bias,[(0,0),(1,0),(0,0),(0,0),(0,0)],
                                              constant_values=jnp.finfo(past_attention_bias.dtype).min)
                xk = jnp.concatenate((past_xk, xk), axis = -3)
                xv = jnp.concatenate((past_xv, xv), axis = -3)
                attention_bias = jnp.concatenate((past_attention_bias, attention_bias), axis = -1)

            if self.config.add_null_attn:
                xv, xk, attention_bias = self.concat_null_kv(xv, xk, attention_bias)

            # xq = (1, 1024, 32, 128)
            # xk = (1, 1024, 32, 128)
            # attention_bias = (1, 1, 1024, 1024)
            attn_weights = dot_product_attention_weights(
                xq,
                xk,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                deterministic=deterministic,
                dtype=jnp.promote_types(self.dtype, jnp.float32),
                precision=self.precision,
            )
            print(f"{attn_weights.shape=}")
            self.sow('intermediates', 'attn_weights', attn_weights)
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
            if sliding_window and n_windows>1:
                attn_output = rearrange(attn_output, 
                        'b s l ... -> b (s l) ...',s=n_windows)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
    
    @jax.profiler.annotate_function
    def concat_null_kv(self, xv, xk, attention_bias):
        
        attention_bias_shape = np.array(attention_bias.shape)
        attention_bias_shape[-1] = 1
        xk_shape = np.array(xk.shape)
        xk_shape[-3] = 1
        
        null_k = jnp.broadcast_to(self.null_k, xk_shape)
        null_v = jnp.broadcast_to(self.null_v, xk_shape)
        xk = jnp.concatenate((xk, null_k), axis = -3)
        xv = jnp.concatenate((xv, null_v), axis = -3)
        attention_bias = jnp.concatenate((attention_bias, jnp.full(attention_bias_shape, 0.0)), axis = -1)
        return xv, xk, attention_bias
        

class FlaxRPTCrossAttention(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.wq = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.wk = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.wv = nn.Dense(
            config.num_attention_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.wo = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )

        # self.resid_dropout = nn.Dropout(rate=config.resid_pdrop,broadcast_dims=(0,))

        if self.config.rot_dim is not None and self.config.rot_dim>0:
            rot_dim = self.config.rot_dim
        else:
            rot_dim = self.head_dim
            
        self.freqs_cis = precompute_freqs_cis(
            rot_dim,
            config.max_sequence_length * 2,
            dtype=self.dtype,
        )
        self.null_k =  self.param(f'null_k', jax.nn.initializers.normal(0.0001), (1,1,self.num_heads,self.head_dim))
        self.null_v =  self.param(f'null_v', jax.nn.initializers.normal(0.0001), (1,1,self.num_heads,self.head_dim))
        
    @jax.profiler.annotate_function
    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))
    @jax.profiler.annotate_function
    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
    @jax.profiler.annotate_function
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        key_value_states: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        kv_position_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        
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
            position_ids = jnp.arange(query_length, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(position_ids[None, :], (batch_size, query_length))
            
        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
        if not is_cross_attention:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=self.dtype, rot_dim=self.config.rot_dim)
        else:
            if kv_position_ids is None:
                kv_position_ids = jnp.arange(key_length, dtype=jnp.int32)
                kv_position_ids = jnp.broadcast_to(kv_position_ids[None, :], (batch_size, key_length))
            freqs_cis_k = jnp.take(self.freqs_cis, kv_position_ids, axis=0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k, dtype=self.dtype, rot_dim=self.config.rot_dim)

        null_k = jnp.broadcast_to(self.null_k, (batch_size, 1, self.num_heads, self.head_dim))
        xk = jnp.concatenate((xk, null_k), axis = -3)
        
        
        null_v = jnp.broadcast_to(self.null_v, (batch_size, 1, self.num_heads, self.head_dim))
        xv = jnp.concatenate((xv, null_v), axis = -3)
        
                
        if attention_mask is not None:
            
            null_mask = jnp.ones((attention_mask.shape[0], 1), dtype=jnp.float32)
            attention_mask = jnp.concatenate((attention_mask, null_mask), axis = -1)
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

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
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=self.precision,
        )
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        # attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
    


    

class FlaxRPTMLP(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        config = self.config

        self.w1 = nn.Dense(
            config.intermediate_size if config.gated_ff else 4*config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.w2 = nn.Dense(
            config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        if self.config.gated_ff:
            self.w3 = nn.Dense(
                config.intermediate_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                use_bias=False,
                kernel_init=dense_init(self.config),
                precision=self.precision,
            )
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop,broadcast_dims=(0,))
    @jax.profiler.annotate_function
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.config.gated_ff:
            x1 = nn.silu(self.w1(x))
            x3 = self.w3(x)
            if self.config.mult_in_complex:
                x =  mult_in_complex(x1,x3)
            else:
                x = x1*x3
            x = self.w2(x)
                
            x = self.dropout(x, deterministic=deterministic)
        else:
            x = nn.gelu(self.w1(x))
            x = self.dropout(x, deterministic=deterministic)
            x = self.w2(x)

        return x

# from transformers.modeling_flax_utils import 
from transformers.generation.flax_logits_process import FlaxLogitsProcessorList
from transformers.generation.flax_utils import SampleState

class FlaxRPTPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RPTConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: RPTConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        print("inside init_cache")
        input_ids = jnp.ones((batch_size, max_length))
        # attention_mask = jnp.zeros_like(input_ids)
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
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

        batch_size, sequence_length = input_ids.shape

        # if position_ids is None:
        #     if past_key_values is not None:
        #         raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

        #     position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is not None:
            attention_mask = jnp.array(attention_mask, dtype="i4")
            # attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            input_ids=jnp.array(input_ids, dtype="i4"),
            attention_mask=attention_mask,
            encoded_neighbors=encoded_neighbors,
            # position_ids=jnp.array(position_ids, dtype="i4"),
            deterministic=not train,
            retriever_supervision=None,
            train_step=None,
            init_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs
    
    def preret_forward(
            self,
            hidden_states,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            output_attentions:bool = False,
            dropout_rng = None,
        ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng)

            
        outputs = self.module.apply(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                deterministic= not train,
                output_attentions=output_attentions,
                method=self.module._encode_forward,
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
            output_attentions:bool = False,
            dropout_rng = None,
        ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng, past_key_values)

        
        outputs, past_key_values = self.module.apply(
                hidden_states=hidden_states,
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask,
                deterministic=not train,
                output_attentions=output_attentions,
                method=self.module._augment_forward,
                **apply_kwargs
        )

        return outputs, unfreeze(past_key_values["cache"]), past_key_values.get("intermediates",None)
    
    def lowcoder_forward(
            self,
            input_ids,
            attention_mask=None,
            params: dict = None,
            train: bool = False,
            past_key_values: dict = None,
            output_attentions:bool = False,
            dropout_rng = None,
        ):

        apply_kwargs = self.create_apply_kwargs(params, dropout_rng, past_key_values)


        outputs, past_key_values = self.module.apply(
                input_ids=input_ids,
                attention_mask=attention_mask,
                deterministic=not train,
                output_attentions=output_attentions,
                method=self.module._lowcoder_forward,
                **apply_kwargs       
        )
        return outputs, unfreeze(past_key_values["cache"]), past_key_values.get("intermediates",None)
    


    def create_apply_kwargs(self, params, dropout_rng, past_key_values=None):
        rngs = {}
        # Handle any PRNG if needed
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        variables = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache
        # has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable
        # so that it can be changed by FlaxGPTJAttention module
        if past_key_values is not None:
            variables["cache"] = past_key_values
            mutable = ["cache","intermediates"]
        else:
            mutable = False
        return dict(rngs=rngs, variables=variables, mutable=mutable)


    #almost the same as the _sample in FlaxGenerationMixin, except we return SampleState instead of FlaxSampleOutput
    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

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
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token * ~next_is_sent_finished + pad_token_id * next_is_sent_finished
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)
        # state = lax.cond(model_kwargs['attention_mask'].sum() > 1, lambda: sample_search_body_fn(state), lambda: state )

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)
            
        past_key_values = state.model_kwargs['past_key_values']
        last_lowcoder_states = past_key_values['transformer']['cached_array']
        
        encoded_lowcoder_states = self.preret_forward(
                           hidden_states=last_lowcoder_states,
                           attention_mask = jnp.ones(last_lowcoder_states.shape[:-1]),
                           params=params)
        
        return state, encoded_lowcoder_states
    

class FlaxRPTLowcoderLayer(nn.Module):
    """

    """
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self) -> None:
        attention_module = FlaxRPTAttention
        if self.config.remat_attention != '':
            # E: repeat function a ton of times for some reason
            attention_module = remat(
                FlaxRPTAttention, static_argnums=(3, 4, 5,-1),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)
            )
        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        mlp_module = FlaxRPTMLP
        if self.config.remat_mlp != '':
            mlp_module = remat(
                FlaxRPTMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp)
            )

        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)

    @jax.profiler.annotate_function
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        fcm_mask: Optional[jnp.ndarray] = None,
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

    

class FlaxRPTLowcoderLayerCollection(nn.Module):
    """
    Basic series of masked attention encoders
    """
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        block = FlaxRPTLowcoderLayer
        if self.config.remat_block != '':
            block = remat(
                FlaxRPTLowcoderLayer, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        assert (self.config.num_hidden_layers%2) == 0, f"config.num_hidden_layers should be devisible by 2"
        num_hidden_layers = self.config.num_hidden_layers//2
        print("In Lowcoder: Using {} layers".format(num_hidden_layers))
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            ) for i in range(num_hidden_layers)
        ]
    @jax.profiler.annotate_function
    def __call__(
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


class FlaxRPTLowcoder(nn.Module):
    """
    Just a bunch of attention layers
    """
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32 # type of embedding
    param_dtype: jnp.dtype=jnp.float32 # type of input
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):        
        self.layers = FlaxRPTLowcoderLayerCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        
    @jax.profiler.annotate_function
    def __call__(
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






class FlaxRPTChunkedCrossAttention(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.chunk_size = self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.cross_attention  = FlaxRPTCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
    @jax.profiler.annotate_function
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        neighbor_hidden_states,
        neighbor_mask,
        position_ids: Optional[jnp.array] = None,
        output_attentions: bool=False,
        deterministic:bool=True,
    ) -> Tuple[jnp.ndarray]:
        
        chunk_size = self.chunk_size
        causal_padding = chunk_size - 1
        num_devices, seq_len, hidden_dim = hidden_states.shape
        num_document_chunks, num_neighbors, _, _ = neighbor_hidden_states.shape
        
        # -> (-1, num_devices_chunks, num_neighbors, 2*chunk_size, hidden_dim)
        neighbor_hidden_states = neighbor_hidden_states.reshape([-1, 2*chunk_size*num_neighbors, hidden_dim])
        neighbor_mask = neighbor_mask.reshape([-1, 2*chunk_size*num_neighbors])
        local_device_count = hidden_states.shape[0]
        if num_document_chunks>1:
            num_devices_chunks = num_document_chunks//local_device_count
            # ->  (-1 ,chunk_size, hidden_dim)
            hidden_states = hidden_states.reshape([-1, num_devices_chunks*chunk_size, hidden_dim])
            hidden_states = jnp.pad(hidden_states[:,causal_padding:,:], ((0,0),(0, causal_padding),(0,0)), 'constant')
            hidden_states = hidden_states.reshape([-1,chunk_size, hidden_dim])
            
            position_ids = jnp.arange(chunk_size)+chunk_size-1
            position_ids = jnp.broadcast_to(position_ids[None, :], (hidden_states.shape[0], chunk_size))
        else:
            hidden_states = hidden_states.reshape([1,1, hidden_dim])
            assert position_ids is not None

            
        
        kv_position_ids = jnp.arange(2*chunk_size)
        kv_position_ids = jnp.broadcast_to(kv_position_ids[None, :], (num_document_chunks*num_neighbors, 2*chunk_size))
        kv_position_ids = kv_position_ids.reshape([-1, 2*chunk_size*num_neighbors])
        
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
        if num_document_chunks>1:
            cross_attention_out = cross_attention_out.reshape([-1, num_devices_chunks*chunk_size, hidden_dim])
            # # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)
            cross_attention_out = jnp.pad(cross_attention_out, ((0,0),(causal_padding, 0),(0,0)), 'constant')[:,:-causal_padding]
        cross_attention_out = cross_attention_out.reshape([num_devices, seq_len, hidden_dim])        
        return (cross_attention_out,)+output[1:]
    

class FlaxRPTUpcoderLayer(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None
    has_cca: bool = False

    def setup(self) -> None:
        attention_module = FlaxRPTAttention
        if self.config.remat_attention != '':
            attention_module = remat(
                FlaxRPTAttention, static_argnums=(3, 4, 5,-1),
                policy=get_gradient_checkpoint_policy(self.config.remat_attention)
            )
        if self.has_cca:
            self.cca = FlaxRPTChunkedCrossAttention(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
            )
            self.cca_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
            if self.config.use_cca_norm2:
                self.cca_norm2 = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
            else:
                self.cca_norm2 = None
                
        else:
            self.cca = None
            self.cca_norm = None
        
        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        mlp_module = FlaxRPTMLP
        if self.config.remat_mlp != '':
            mlp_module = remat(
                FlaxRPTMLP, static_argnums=(1,),
                policy=get_gradient_checkpoint_policy(self.config.remat_mlp)
            )

        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
        )
        self.attention_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.ffn_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
    @jax.profiler.annotate_function
    def __call__(
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
        fcm_mask: Optional[jnp.ndarray] = None,
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
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        block = FlaxRPTUpcoderLayer
        if self.config.remat_block != '':
            block = remat(
                FlaxRPTUpcoderLayer, static_argnums=(6, 7, 8, 9),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        assert (self.config.num_hidden_layers%2) == 0, f"config.num_hidden_layers should be divisible by 2"
        num_hidden_layers = self.config.num_hidden_layers//2
        print("In Upcoder: Using {} layers".format(num_hidden_layers))
        def has_cca(layer_index):
            if self.config.cca_freq is None or self.config.cca_freq == 0:
                return False
            return (layer_index % self.config.cca_freq) == 0  
            
            
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                has_cca = has_cca(i),
            ) for i in range(num_hidden_layers)
        ]
    @jax.profiler.annotate_function
    def __call__(
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
                hidden_states, #0
                attention_mask, #1
                position_ids, #2
                neighbor_hidden_states, #3
                neighbor_mask, #4
                chunk_index, #5
                deterministic, #6
                init_cache, #7
                output_attentions, #8
                None, #fcm_mask= #9
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
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None
    
    def setup(self):
        
        self.postret_bidir_attention = FlaxRPTCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.postret_bi_attention_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.query_nei_xattention_qnorm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.query_nei_xattention_knorm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        
        self.query_nei_xattention = FlaxRPTCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)

    @jax.profiler.annotate_function
    def __call__(self, hidden_states, neighbor_hidden_states, neighbor_mask, output_attentions, deterministic, query_hidden_states=None):
        neighbor_hidden_states_shape = neighbor_hidden_states.shape
        num_document_chunks, num_neighbors, ret_size, hidden_dim = neighbor_hidden_states_shape
        assert ret_size == 2*self.config.chunk_size
        neighbor_hidden_states = neighbor_hidden_states.reshape([num_document_chunks*num_neighbors,
                                                                 ret_size,
                                                                 hidden_dim])
        neighbor_mask = neighbor_mask.reshape([num_document_chunks*num_neighbors,ret_size])
        
        #Non-causal self attention with the two parts of the neighbor chunk
        #For each neighbor we also retrieve it's subsequent chunk, so it helps to have the two parts of the neighbor chunk
        #Attend to each other (it was only causal before)
        postret_bi_output  = self.postret_bidir_attention(
                                                self.postret_bi_attention_norm(neighbor_hidden_states),
                                                attention_mask=neighbor_mask,
                                                deterministic=deterministic,
                                                output_attentions=output_attentions) #need(!!!) to cache this during generation
        neighbor_hidden_states = postret_bi_output[0] + neighbor_hidden_states
        
        #Cross Attention with the neighbor chunk hidden states as Q, and the query chunk hidden states as KV
        if query_hidden_states is None: # if we didn't get it from update_inputs_for_generation
            query_hidden_states = hidden_states
        query_hidden_states = einops.repeat(query_hidden_states.reshape([-1, self.config.chunk_size, hidden_dim]),
                                    'b n d -> (b k) n d', n = self.config.chunk_size, k = num_neighbors)
        assert query_hidden_states.shape[0] == num_document_chunks*num_neighbors

            
        
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
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None
    
    def setup(self):
        self.cross_neig_causal_att = FlaxRPTAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.xnei_norm1 = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.xnei_norm2 = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.weight = nn.Dense(features=1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=dense_init(self.config),
                precision=self.precision,
                use_bias=True)
        if self.config.use_xnei_bias:
            self.xnei_bias = self.param(f'xnei_bias', jax.nn.initializers.normal(0.01), (1,1,1,self.config.hidden_size))
        else:
            self.xnei_bias = None
            
    @jax.profiler.annotate_function
    def __call__(self,
                 neighbor_hidden_states,
                 neighbor_mask,
                 output_attentions,
                 deterministic,
                 device_count):
        num_document_chunks, num_neighbors, ret_size, hidden_dim = neighbor_hidden_states.shape
        if num_document_chunks>1:
            num_devices_chunks = num_document_chunks//device_count
            #this pooled tensor has shape [device_count, num_devices_chunks*num_neighbors, hidden_dim]
        else:
            num_devices_chunks = 1
        new_shape = [device_count, num_devices_chunks*num_neighbors, ret_size, hidden_dim]
        pooled_neighbor_hidden_states = neighbor_hidden_states.reshape(new_shape).mean(axis=-2) 
        
        pooled_neighbor_mask = neighbor_mask.reshape([device_count, num_devices_chunks*num_neighbors, ret_size]).any(axis=-1)
        
        
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
        ret_gate_score = self.weight(pooled_neighbor_hidden_states) # [device_count, num_devices_chunks*num_neighbors, 1]
        ret_gate_score = rearrange(ret_gate_score, 't (c k) 1 -> (t c) k 1 1 ',
                                   t=device_count, c=num_devices_chunks, k=num_neighbors)
        
        if self.xnei_bias is not None:
            neighbor_hidden_states += ret_gate_score*self.xnei_bias
        else:
            ret_gate_score = 0.1+0.9*jax.nn.sigmoid(ret_gate_score/hidden_dim)
            neighbor_hidden_states = ret_gate_score*neighbor_hidden_states
        return (neighbor_hidden_states,) + cross_neig_out[1:]

class FlaxRPTUpcoder(nn.Module):
    """

    """
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):        
        if self.config.augment_neighbors:
            self.neighbor_augmentor = FlaxRPTNeighborAugmentor(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        else:
            self.neighbor_augmentor = None
            
            
        if self.config.augment_across_neighbors:
            self.neighbor_cross_augmentor = FlaxRPTCrossNeighborAugmentor(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        else:
            self.neighbor_cross_augmentor = None
        self.layers = FlaxRPTUpcoderLayerCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        
    def augment(self,
                hidden_states,
                neighbor_hidden_states,
                neighbor_mask,
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
        
    @jax.profiler.annotate_function
    def __call__(
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
        #else We are generating... And have already augmented the neighbor hidden states.

            
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
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.preret_bidir_attention = FlaxRPTCrossAttention(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.preret_bi_attention_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.pre_key_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.key_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.pre_query_norm = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        self.query_projection = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=dense_init(self.config),
            precision=self.precision,
        )
        self.fill_value = self.config.retriever_fill_value
        self.n_skip_chunks = (self.config.max_sequence_length//self.config.n_windows)//self.config.chunk_size
        self.num_neighbors = self.config.num_neighbors
        self.threshold_nei_scores = self.config.threshold_nei_scores
        self.num_sequence_chunks = self.config.num_sequence_chunks
        if self.config.aux_loss_schedule_steps is not None:
            assert self.config.aux_scale is not None
            self.aux_scale = self.config.aux_scale
            self.aux_loss_schedule_fn = optax.linear_schedule(0,1,self.config.aux_loss_schedule_steps)
            
        if self.config.max_margin is not None and self.config.margin_schedule_steps is not None:
            assert self.config.max_margin>=1
            self.increase_margin_schedule_fn = optax.linear_schedule(1,self.config.max_margin, self.config.margin_schedule_steps)
        
        if self.config.ss_schedule_steps is not None and \
                        self.config.scheduled_sampling_max_prob is not None \
                        and self.config.scheduled_sampling_min_prob is not None \
                            and self.has_rng("dropout"):
            self.ss_rng = self.make_rng("dropout")
            self.scheduled_sampling_schedule_fn = m1_cosine_decay_schedule(decay_steps=self.config.ss_schedule_steps,
                                                                            min_value=self.config.scheduled_sampling_min_prob,
                                                                            max_value=self.config.scheduled_sampling_max_prob)
    @jax.profiler.annotate_function
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
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

        query_based_scores = jnp.einsum('qd,kd->qk', encoded_output.query_chunks, 
                                                        encoded_output.key_chunks,
                                                        precision=self.precision)
        query_based_scores /= jnp.sqrt(self.config.hidden_size)
        
        segment_mask = create_segment_mask(query_based_scores.shape[0], self.n_skip_chunks)

        chunk_mask = encoded_output.chunk_mask
        chunk_mask &= segment_mask
        if retriever_supervision is not None:
            aux_loss, target_neighbor_mask, target_score_based_idx, ret_metrics = self.compute_retriever_loss(query_based_scores,
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
                            jnp.take_along_axis(chunk_mask, query_score_based_idx,axis=-1),
                            target_score_based_idx,
                            target_neighbor_mask, 
                            train_step,
                            deterministic)
        neighbor_hidden_states =  self.lookup_neighbor_states(encoded_output.encoded_hidden_states, top_nei_idx)
        nei_mask = jnp.broadcast_to(jnp.expand_dims(nei_mask, axis=-1), neighbor_hidden_states.shape[:-1])
        return FlaxRPTRetrieverNeighborOutput(aux_loss=aux_loss if aux_loss is not None else None,
                                              loss_scale=self.get_loss_scale(train_step) if aux_loss is not None else None,
                                              neighbor_hidden_states=neighbor_hidden_states,
                                              neighbor_mask=nei_mask,
                                              retrieval_metrics=jax.tree_map(lambda x: x.mean(), ret_metrics) if ret_metrics is not None else None, 
                                              )
    
    
    # @classmethod
    @jax.profiler.annotate_function
    def lookup_neighbor_states(cls, cand_hidden_states, top_nei_idx):
        num_document_chunks = top_nei_idx.shape[0]
        shifted_hidden_states = jnp.pad(cand_hidden_states[1:,...],((0,1),(0,0),(0,0)))
        curr_neighbor_hidden_states = cand_hidden_states[top_nei_idx.reshape(-1)]
        next_neighbor_hidden_states = shifted_hidden_states[top_nei_idx.reshape(-1)]
        neighbor_hidden_states = jnp.concatenate((curr_neighbor_hidden_states, next_neighbor_hidden_states), axis=-2)
        neighbor_hidden_states = einops.rearrange(neighbor_hidden_states, '(b k) r d -> b k r d', b=num_document_chunks)
        return neighbor_hidden_states
    
    @jax.profiler.annotate_function
    def preret_encode(self,
               hidden_states,
               attention_mask,
               deterministic,
               output_attentions: bool = False,):
        original_hidden_states_shape = hidden_states.shape
        original_attention_mask_shape = attention_mask.shape
        
        original_hidden_states, attention_mask = jax.tree_map(
                lambda x: einops.rearrange(x, 'b (l c) ... -> (b l) c ... ', c=self.config.chunk_size), 
                (hidden_states, attention_mask)
                ) # add a chunk dimension
        #1. apply bi-dir attention 
        preret_bi_output  = self.preret_bidir_attention(
                                                self.preret_bi_attention_norm(original_hidden_states),
                                                attention_mask=attention_mask,
                                                deterministic=deterministic,
                                                output_attentions=output_attentions)
        encoded_hidden_states = preret_bi_output[0] + original_hidden_states
        
        #2. pool
        pooled_hidden_states = encoded_hidden_states.mean(axis=-2)
        
        #3. project to query chunks and key chunks
        key_chunks = self.key_projection(self.pre_key_norm(pooled_hidden_states))
        query_chunks = self.query_projection(self.pre_query_norm(pooled_hidden_states))
        chunk_mask = attention_mask.astype(bool).any(-1)[...,None]
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
    @jax.profiler.annotate_function
    def apply_scheduled_sampling(self,
                              query_score_based_idx,
                              chunk_mask,
                              target_score_based_idx,
                              target_neighbor_mask, 
                              train_step,
                              deterministic):
        if deterministic or self.is_initializing() or target_score_based_idx is None:
            top_nei_idx,top_nei_mask = query_score_based_idx, chunk_mask
        else:
                
            rv = jax.random.bernoulli(key=self.ss_rng,
                                p=self.scheduled_sampling_schedule_fn(train_step if not self.is_initializing() else 1),
                                shape=()) #this is a boolean of shape [1]
            top_nei_idx, top_nei_mask = jax.lax.cond(rv,
                                    (), lambda args: (query_score_based_idx, chunk_mask),
                                    (),  lambda args: (target_score_based_idx, target_neighbor_mask)
                                    )
        return top_nei_idx, top_nei_mask

    @jax.profiler.annotate_function
    def compute_retriever_loss(self, raw_query_scores, retriever_supervision, chunk_mask, train_step):        
        def f(x):
            return x.reshape((-1,self.config.num_scored_neighbors))
        retriever_supervision = jax.tree_map(f, retriever_supervision)
        
        nei_idx = retriever_supervision.nei_idx #[num_sequence_chunks, num_scored_neighbors]
        nei_scores = retriever_supervision.nei_scores
        
        raw_target_scores = create_target_scores(raw_query_scores, nei_idx, nei_scores, fill_value=self.fill_value)
        raw_target_scores_wz = create_target_scores(raw_query_scores, nei_idx, nei_scores, fill_value=0)
        
        threshold_mask = self.threshold_nei_scores<raw_target_scores
        allowed_neighbor_mask = combine_masks(threshold_mask, chunk_mask,dtype=bool) #allowed neighbors
        margin = self.increase_margin_schedule_fn(train_step if not self.is_initializing() else 1)
        pair_loss = jax.nn.relu(margin - compute_pairs(raw_query_scores, operator.sub)) # [num_document_chunks, num_document_chunks*num_document_chunks]
        
        #one of the scores needs to be above the threshold for the pair to be valid
        valid_pairs =  combine_masks(compute_pairs(raw_target_scores, lambda x,y: x>y),
                                    compute_pairs(threshold_mask, lambda x, y: x),
                                    compute_pairs(chunk_mask, operator.and_)
                                    )
    
        pair_loss  = jnp.where(valid_pairs, pair_loss, 0.0)
        if self.config.return_ret_metrics:
            metrics = compute_retrieval_metrics(raw_query_scores, raw_target_scores_wz,
                                           query_mask=chunk_mask,
                                           target_mask=allowed_neighbor_mask)
        else:
            metrics = None
        ndcg_lambda = compute_ndcg_lambda(raw_query_scores, raw_target_scores_wz,
                                           query_mask=chunk_mask,
                                           target_mask=allowed_neighbor_mask,)
        
        per_chunk_pair_loss = (ndcg_lambda*pair_loss).sum(axis=-1)
        
        any_mask = combine_masks(threshold_mask.any(axis=-1), chunk_mask.any(axis=-1),dtype=bool)
        raw_aux_loss = jnp.where(any_mask, per_chunk_pair_loss, 0.0).sum()
        
        
        target_idx = topk_chunks(raw_target_scores, num_candidates=self.num_neighbors, where=chunk_mask)
        target_nei_mask = jnp.take_along_axis(allowed_neighbor_mask, target_idx,axis=-1)
        return (raw_aux_loss, valid_pairs.sum(), ), target_nei_mask, target_idx,metrics
        
    
    def get_loss_scale(self,train_step):
        loss_scale = self.aux_loss_schedule_fn(train_step if not self.is_initializing() else 1)
        return loss_scale*self.aux_scale
#loss is calculated as lm_loss + (raw_aux_loss/valid_pairs.sum())* self.get_loss_scale(train_step)


class FlaxRPTModule(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        # the embedding dim
        self.embed_dim = self.config.hidden_size

        #TODO: move this wte and dropout into the lowcoder.

        # define a dropout layer
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)

        # word to embedding module (layer)
        self.wte = nn.Embed(
            self.config.vocab_size, # input size
            self.config.hidden_size, # embedding size
            embedding_init=dense_init(self.config, is_embedding=True), # basically np.random of weights in the correct size
            dtype=self.dtype, # type of embedding vector entries
            param_dtype=self.param_dtype, # type of input
        )
        
        self.lowcoder = FlaxRPTLowcoder(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        if self.config.cca_freq is not None and self.config.cca_freq > 0:
            self.retriever = FlaxRPTRetriever(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        else:
            self.retriever = None
            
        self.upcoder = FlaxRPTUpcoder(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        
        #TODO: move this ln_f into the upcoder.
        self.ln_f = FlaxRPTRMSNorm(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        

    @nn.compact
    def _concatenate_to_lowcoder_cache(self, array):
        chunk_size = self.config.chunk_size
        is_initialized = self.has_variable("cache", "cached_array")
        *batch_dims, _, hidden_dim = array.shape
        cached_array = self.variable("cache", "cached_array",
                                     jnp.zeros,
                                     tuple(batch_dims)+(self.config.chunk_size, hidden_dim),
                                     array.dtype)
        if is_initialized:
            last_chunk = array[...,-chunk_size:,:]
            
            num_updated_cache_vectors = last_chunk.shape[-2]
            shift = self.config.chunk_size-num_updated_cache_vectors #will need to update if I change retrieval stride                        
            indices = (0,) * len(batch_dims) + (shift,  0)

            array_operand = jnp.roll(cached_array.value, shift=-num_updated_cache_vectors, axis=-2)
            cached_array.value = lax.dynamic_update_slice(array_operand,
                                        last_chunk,
                                        indices)
    @jax.profiler.annotate_function
    def __call__(
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
        upcoder_input = None,
        encoded_neighbors: Optional[EncodedNeighbors] = None,
    ):
        lowcoder_outputs = None
        retriever_output = None
        neighbor_hidden_states = None
        neighbor_mask = None
        chunk_index = None          
        retriever_input = None

        
        if upcoder_input is None:
            input_embeds = self.wte(input_ids.astype("i4"))

            hidden_states = self.dropout(input_embeds, deterministic=deterministic)

            lowcoder_outputs = self.lowcoder(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            

            hidden_states = lowcoder_outputs.last_hidden_state if  return_dict else lowcoder_outputs[0] 
            if self.has_variable("cache", "cached_array") or init_cache:
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
        
        
        hidden_states = upcoder_outputs.last_hidden_state if  return_dict else upcoder_outputs[0] 
        hidden_states = self.ln_f(hidden_states)
        

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

@add_start_docstrings("", "")
class FlaxRPTModel(FlaxRPTPreTrainedModel):
    module_class = FlaxRPTModule


class FlaxRPTForCausalLMModule(nn.Module):
    config: RPTConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None
    
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
            self.transformer.wte(input_ids.astype("i4")),
            attention_mask,
            **kwargs
        )

        outputs = self.transformer.retriever.preret_encode(
            hidden_states=lowcoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs

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
    
    
    def setup(self):
        self.transformer = FlaxRPTModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )
    @jax.profiler.annotate_function
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        retriever_supervision: RetrieverSupervision = None,
        train_step: Optional[int] = None,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        upcoder_input = None,
        encoded_neighbors: Optional[EncodedNeighbors] = None,
        
    ):
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        transformer_input = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            upcoder_input=upcoder_input,
            encoded_neighbors=encoded_neighbors,
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
        if jax.process_count() > 1:
            transformer = jax.vmap(transformer)
            transformer_input = jax.tree_map(add_process_dim, transformer_input)
            outputs = transformer(**transformer_input)
            outputs = jax.tree_map(remove_process_dim, outputs)
        else:
            outputs = transformer(**transformer_input)

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0] 


        lm_logits = self.unembed(hidden_states) 
            

        if not return_dict:
            return (lm_logits,) + outputs[1:]
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
        )
    @jax.profiler.annotate_function
    def unembed(self, hidden_states):
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if self.config.palm_init:
            lm_logits = lm_logits/jnp.sqrt(hidden_states.shape[-1])
        return lm_logits



@add_start_docstrings("", "")
class FlaxRPTForCausalLM(FlaxRPTPreTrainedModel):
    module_class = FlaxRPTForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids,
                                      max_length,
                                      attention_mask = None,
                                      encoded_neighbors=None,
                                      past_key_values=None,
                                      ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        if past_key_values is None:
            past_key_values = self.init_cache(batch_size, self.config.window_length)

        return {
            "past_key_values": past_key_values,
            "encoded_neighbors": encoded_neighbors,
            "attention_mask": attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        
        if model_kwargs.get("encoded_neighbors", None) is not None:
            encoded_neighbors = model_kwargs["encoded_neighbors"]
            if encoded_neighbors.chunk_index is None:
                chunk_index = jnp.zeros([1,1],dtype=jnp.int32) #this assumes bs=1
            else:
                chunk_index = jnp.clip(encoded_neighbors.chunk_index + 1, a_max=self.config.chunk_size-1) #will need to modify later
            
            encoded_neighbors = EncodedNeighbors(
                            neighbor_hidden_states=encoded_neighbors.neighbor_hidden_states[-1:,...], #assumes bs=1
                            neighbor_mask=encoded_neighbors.neighbor_mask[-1:,...], #assumes bs=1
                            chunk_index=chunk_index,
                            )
            model_kwargs["encoded_neighbors"]  = encoded_neighbors
        model_kwargs['attention_mask'] = jnp.ones([1,1],dtype=jnp.int32)
        return model_kwargs