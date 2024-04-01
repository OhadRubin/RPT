import mlxu

from EasyLM.models.rpt.rpt_model import RPTConfig
import rpt_model
import rpt_model_torch
import torch
import jax.numpy as jnp
import numpy as np
import jax
import flax


rpt_config = RPTConfig(
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
    add_outputs=False,
    single_model=True,
    nearest_chunk_distance=16,
    override_list="",
    max_new_tokens=2048,
    iterative_mode=True,
    dense_mem=True,
    num_neighbors=2,
)

def test_rms_norm():
    flax_rms_norm = rpt_model.FlaxRPTRMSNorm(rpt_config, dtype=jnp.float32, param_dtype=jnp.float32)
    torch_rms_norm = rpt_model_torch.TorchRPTRMSNorm(rpt_config, dtype=torch.float32, param_dtype=torch.float32)

    arr = np.zeros(4096)
    arr[:3] = [1, 2, 3]

    torch_result = torch_rms_norm.forward(torch.Tensor(arr))
    flax_result = flax_rms_norm.init_with_output(jax.random.PRNGKey(42), jnp.array(arr))

    np.testing.assert_almost_equal(torch_result.numpy(), flax_result[0])

def test_attention():
    input_id = np.ones(1024, dtype=int)
    input_id[:3] = [1, 2, 3]

    wte = flax.linen.Embed(
        rpt_config.vocab_size,  # input size
        rpt_config.hidden_size,  # embedding size
        embedding_init=rpt_model.dense_init(rpt_config, is_embedding=True),  # basically np.random of weights in the correct size
        dtype=jnp.float32,  # type of embedding vector entries
        param_dtype=jnp.float32,  # type of input
    )

    embedded_input = wte.init_with_output(jax.random.PRNGKey(42), input_id)

    flax_rpt_model = rpt_model.FlaxRPTAttention(rpt_config, dtype=jnp.float32, param_dtype=jnp.float32)
    torch_rpt_model = rpt_model_torch.TorchRPTAttention(rpt_config, dtype=torch.float32)

    flax_result = flax_rpt_model.init_with_output(jax.random.PRNGKey(42), jnp.array([embedded_input[0]]), jnp.ones_like(input_id))
    torch_result = torch_rpt_model.forward(torch.Tensor(np.array([embedded_input[0]])), torch.ones_like(torch.Tensor(input_id)))

    np.testing.assert_almost_equal(torch_result[0].detach().numpy(), flax_result[0][0], decimal=0)


def test_cross_attention():
    input_id = np.ones(1024, dtype=int)
    input_id[:3] = [1, 2, 3]

    wte = flax.linen.Embed(
        rpt_config.vocab_size,  # input size
        rpt_config.hidden_size,  # embedding size
        embedding_init=rpt_model.dense_init(rpt_config, is_embedding=True),
        # basically np.random of weights in the correct size
        dtype=jnp.float32,  # type of embedding vector entries
        param_dtype=jnp.float32,  # type of input
    )

    embedded_input = wte.init_with_output(jax.random.PRNGKey(42), input_id)

    flax_rpt_model = rpt_model.FlaxRPTCrossAttention(rpt_config, dtype=jnp.float32, param_dtype=jnp.float32)
    torch_rpt_model = rpt_model_torch.TorchRPTCrossAttention(rpt_config, dtype=torch.float32)

    flax_result = flax_rpt_model.init_with_output(jax.random.PRNGKey(42), jnp.array([embedded_input[0]]),
                                                  jnp.array([embedded_input[0]]))
    torch_result = torch_rpt_model.forward(torch.Tensor(np.array([embedded_input[0]])), torch.Tensor(np.array([embedded_input[0]])))

    np.testing.assert_almost_equal(torch_result[0].detach().numpy(), flax_result[0][0], decimal=5)


def test_mlp():
    input_id = np.ones(1024, dtype=int)
    input_id[:3] = [1, 2, 3]

    wte = flax.linen.Embed(
        rpt_config.vocab_size,  # input size
        rpt_config.hidden_size,  # embedding size
        embedding_init=rpt_model.dense_init(rpt_config, is_embedding=True),  # basically np.random of weights in the correct size
        dtype=jnp.float32,  # type of embedding vector entries
        param_dtype=jnp.float32,  # type of input
    )

    embedded_input = wte.init_with_output(jax.random.PRNGKey(42), input_id)

    flax_rpt_model = rpt_model.FlaxRPTMLP(rpt_config, dtype=jnp.float32, param_dtype=jnp.float32)
    torch_rpt_model = rpt_model_torch.TorchRPTMLP(rpt_config, dtype=torch.float32)

    flax_result = flax_rpt_model.init_with_output(jax.random.PRNGKey(42), jnp.array([embedded_input[0]]))
    torch_result = torch_rpt_model.forward(torch.Tensor(np.array([embedded_input[0]])))

    np.testing.assert_almost_equal(torch_result[0].detach().numpy(), flax_result[0][0], decimal=0)


def test_lowcoder():
    input_ids = np.ones(1024, dtype=int)
    input_ids[:3] = [1, 2, 3]
    seq_length = input_ids.shape[0]
    batch_size = 1
    attention_mask = jnp.ones_like(input_ids)
    position_ids = jnp.broadcast_to(
        jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
        (batch_size, seq_length)
    )

    wte = flax.linen.Embed(
        rpt_config.vocab_size,  # input size
        rpt_config.hidden_size,  # embedding size
        embedding_init=rpt_model.dense_init(rpt_config, is_embedding=True),  # basically np.random of weights in the correct size
        dtype=jnp.float32,  # type of embedding vector entries
        param_dtype=jnp.float32,  # type of input
    )

    hidden_states = wte.init_with_output(jax.random.PRNGKey(42), input_ids)

    lowcoder = rpt_model.FlaxRPTLowcoder(rpt_config, dtype=jnp.float32)

    """
    lowcoder_outputs = lowcoder.init_with_output(jax.random.PRNGKey(42),
        jnp.array([hidden_states[0]]),
        attention_mask,
        position_ids=position_ids,
        deterministic=True,
        init_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )

    print(lowcoder_outputs)
    """

    lowcoder_torch = rpt_model_torch.TorchRPTLowcoderLayer(rpt_config, dtype=torch.float32)
    lowcoder_torch_output = lowcoder_torch.forward(
        torch.Tensor(np.array([hidden_states[0]])),
        torch.Tensor(np.array(attention_mask)),
        position_ids=torch.Tensor(np.array(position_ids)),
        deterministic=True,
        init_cache=False,
        output_attentions=False,
    )

    print(lowcoder_torch_output)


test_lowcoder()