from typing import Optional
import torch

def dot_product_attention_weights(
  query: torch.Tensor,
  key: torch.Tensor,
  bias: Optional[torch.Tensor] = None,
  mask: Optional[torch.Tensor] = None,
  broadcast_dropout: bool = True,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[torch.dtype] = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable via
      ``mutable=['intermediates']`` in order to have that collection returned.
      If ``module`` is None, the attention weights will not be sowed.

  Returns:
    Output of shape ``[batch..., num_heads, q_length, kv_length]``.
  """
  query, key = query.type(dtype=dtype), key.type(dtype=dtype)

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / torch.sqrt(torch.tensor(depth)).type(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = torch.einsum(
    '...qhd,...khd->...hqk', query, key
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = torch.finfo(dtype).min
    attn_weights = torch.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = torch.softmax(attn_weights, dim=-1).type(dtype)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = torch.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = torch.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.type(dtype) / torch.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights
