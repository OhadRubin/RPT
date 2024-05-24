import torch

def make_attention_mask(query_input: torch.Tensor,
                        key_input: torch.Tensor,
                        pairwise_fn=torch.multiply,
                        extra_batch_dims=0,
                        dtype=torch.bool):
    mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
    mask = mask.unsqueeze(-3)
    for dim in range(extra_batch_dims):
        mask = mask.unsqueeze(dim)
    return mask.type(dtype)


def make_causal_mask(x,
                     extra_batch_dims=0,
                     dtype=torch.bool):
    idxs = torch.broadcast_to(torch.arange(x.shape[-1]), x.shape)
    return make_attention_mask(idxs, idxs, torch.greater_equal, extra_batch_dims=extra_batch_dims, dtype=dtype)


def combine_masks(*masks, dtype=torch.bool):
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None

    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = torch.logical_and(mask, other_mask)

    return mask.type(dtype)

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))

def silu(x):
    return x * torch.sigmoid(x)

def assign_slice(operand, update, start_indexes):
    slice_indicies = []

    start_indexes = torch.clamp(
        torch.Tensor(start_indexes),
        torch.zeros(len(operand.shape)),
        torch.tensor(operand.shape) - torch.tensor(update.shape)
    ).type(torch.int)
    end_indexes = (start_indexes + torch.tensor(update.shape)).type(torch.int)
    for start_index, end_index in zip(start_indexes, end_indexes):
        slice_indicies.append(slice(start_index, end_index))

    operand[tuple(slice_indicies)] = update
    return operand