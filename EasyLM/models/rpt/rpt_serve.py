import os
if "DEBUG" in os.environ:
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  
  
import pprint
from functools import partial
import numpy as np
import mlxu
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import flax
import optax
from transformers import GenerationConfig, FlaxLogitsProcessorList
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.rpt.rpt_model import RPTConfig, FlaxRPTForCausalLM, FlaxRPTLowcoderRetrieverEncodedOutput, EncodedNeighbors
from EasyLM.models.rpt.memory import Memory
import gin
import tqdm
import absl
from EasyLM.jax_utils import flatten_tree,max_pooling,print_attention_from_intermediates

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
    jax_distributed=JaxDistributedConfig.get_default_config(),
    add_outputs=False,
    single_model=True,
    nearest_chunk_distance=16,
    override_list="",
    max_new_tokens=2048,
    iterative_mode=True,
    dense_mem=True,
    num_neighbors=2,
)

from transformers import AutoTokenizer
import numpy as np
from flax.linen import combine_masks, make_causal_mask
from einops import reduce


def apply_forward_upcoder(params,
                          hf_model,
                          input_tokens,
                          input_mask,
                          output_tokens,
                          output_mask,
                          upcoder_input
                          ):
    outputs, past_key_values = hf_model.module.apply(
        params, input_tokens, attention_mask=input_mask,
        upcoder_input=upcoder_input,
        deterministic=True,
        mutable=['cache','intermediates']
    )
    output = process_logits(output_tokens, output_mask, outputs.logits)
    past_key_values = unfreeze(past_key_values).get("cache", None)
    return output, past_key_values

def apply_forward_loglikelihood(params,
                                hf_model,
                                input_tokens,
                                input_mask,
                                output_tokens,
                                output_mask,
                          ):
    outputs, past_key_values = hf_model.module.apply(
        params, input_tokens, attention_mask=input_mask,
        deterministic=True,
        mutable=['cache']
    )
    output = process_logits(output_tokens, output_mask, outputs.logits)
    past_key_values = unfreeze(past_key_values).get("cache", None)

    return output, past_key_values

def apply_forward_lowcoder(params, hf_model,input_tokens,input_mask, **kwargs):
    outputs, past_key_values = hf_model.module.apply(
                params,
                input_ids=input_tokens,
                attention_mask=input_mask,
                deterministic=True,
                method=hf_model.module._lowcoder_forward,
                mutable = ["cache"]
        )
    past_key_values = unfreeze(past_key_values).get("cache", None)
    return outputs,past_key_values

def apply_forward_augment(params, hf_model, hidden_states, neighbor_hidden_states, neighbor_mask):
    outputs, past_key_values = hf_model.module.apply(
                params,
                hidden_states=hidden_states,
                neighbor_hidden_states=neighbor_hidden_states,
                neighbor_mask=neighbor_mask,
                deterministic=True,
                method=hf_model.module._augment_forward,
                mutable = ["cache"]
        )
    past_key_values = unfreeze(past_key_values).get("cache", None)
    return outputs, past_key_values


def filter_(intermediates):
    def cont(k):
        for s in ["layers/0","retriever","/neighbor_"]:
            if s in k:
                return True
        return False
        
    return {k:v for k,v in intermediates.items() if cont(k)}
def postproc_output(tokenizer, output, output_text, verbose=False):
    new_output_text = []
    for old_text,text in zip(output_text,list(tokenizer.batch_decode(output))):
        if tokenizer.eos_token in text:
            text = text.split(tokenizer.eos_token, maxsplit=1)[0]
        if verbose:
            print(text,end='') 
        new_output_text.append(old_text+(text,))

    return new_output_text

def process_logits(output_tokens, output_mask, logits):
    loglikelihood = -optax.softmax_cross_entropy_with_integer_labels(
        logits, output_tokens
    )
    loglikelihood = jnp.sum(loglikelihood * output_mask, axis=-1)
    match_count = jnp.sum(
        (jnp.argmax(logits, axis=-1) == output_tokens) * output_mask,
        axis=-1
    )
    total = jnp.sum(output_mask, axis=-1)
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


import copy

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    jax.distributed.initialize()
    set_random_seed(FLAGS.seed)

    prefix_tokenizer = RPTConfig.get_tokenizer(truncation_side='left', padding_side='left')
    tokenizer = RPTConfig.get_tokenizer(truncation_side='right', padding_side='right')

    with jax.default_device(jax.devices("cpu")[0]):
        rpt_config = RPTConfig.load_config(FLAGS.load_rpt_config)
        config = RPTConfig()
        override_dict = {key:getattr(config,key) for key in FLAGS.override_list.split(",")}
        rpt_config.update(override_dict)
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )
        print(rpt_config)
        params = unfreeze(params)

        hf_model = FlaxRPTForCausalLM(
            rpt_config,
            input_shape=(jax.local_device_count(), FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False,
        )
        params['cache'] = unfreeze(hf_model.init_cache(FLAGS.lm_server.batch_size, rpt_config.window_length))
            
    # hf_model.save_pretrained("/home/ohadr/meliad2/hf_model_1", params=params)

    model_ps = match_partition_rules(
        RPTConfig.get_partition_rules(), params
    )
    shard_fns, _ = make_shard_and_gather_fns(
        model_ps, get_float_dtype_by_name(FLAGS.dtype)
    )

    def pjit_func(func):
        @partial(
            pjit,
            in_shardings=(model_ps, PS()),
            out_shardings=(PS(), model_ps['cache'])
        )
        def _inner(params, batch):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
            outputs, past_key_values = func(params, hf_model, **batch)
            return outputs, past_key_values
        return _inner
    
    _forward_upcoder = pjit_func(apply_forward_upcoder)
    _forward_loglikelihood = pjit_func(apply_forward_loglikelihood)
    _forward_lowcoder = pjit_func(apply_forward_lowcoder)
    _forward_augment = pjit_func(apply_forward_augment)
    

    def forward_loglikelihood(params, batch, memory):
        if rpt_config.cca_freq==0:
            outputs, past_key_values = _forward_loglikelihood(params, batch)
            if past_key_values is not None:
                params.update(cache=past_key_values)
            return outputs
        else:
            with mesh:
                outputs, past_key_values = _forward_lowcoder(params, batch)
                params.update(cache=past_key_values)

                neighbor_hidden_states, neighbor_mask, *_ = memory.add(
                                    input_tokens=batch["input_tokens"],
                                    encoded_hidden_states=outputs.encoded_hidden_states,
                                    key_chunks=outputs.key_chunks,
                                    query_chunks=outputs.query_chunks,
                                    )
                batch.update(
                    upcoder_input=FlaxRPTLowcoderRetrieverEncodedOutput(
                            hidden_states=outputs.original_hidden_states,
                            attention_mask=outputs.attention_mask,
                            neighbor_hidden_states=neighbor_hidden_states,
                            neighbor_mask=neighbor_mask
                            )
                )
                outputs, past_key_values = _forward_upcoder(params, batch)
                params.update(cache=past_key_values)
            return  outputs


    def  create_forward_generate(model_ps, max_new_tokens,is_prefix=False,sample=True):
        def _forward_generate(params, rng, batch, temperature):
            batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
            rng_generator = JaxRNG(rng)
            if sample:
                generate_kwargs = dict(
                logits_processor=FlaxLogitsProcessorList(
                    [FlaxTemperatureLogitsWarper(temperature)]
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
                    ))
            else:
                generate_kwargs = dict(generation_config=GenerationConfig(
                    max_new_tokens=FLAGS.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                    ),
                )
            
            output,encoded_lowcoder_states = hf_model.generate(
                batch['input_tokens'],
                attention_mask=batch['input_mask'],
                encoded_neighbors=batch.get("encoded_neighbors",None),
                params=params['params'],
                past_key_values=params.get("cache",None), #passing the initilized cache
                prng_key=rng_generator(),
                **generate_kwargs
            )
            past_key_values = output.model_kwargs['past_key_values']
            sequences = output.sequences[:, batch['input_tokens'].shape[1]:]
            return sequences, rng_generator(), past_key_values, encoded_lowcoder_states
        cache_ps = model_ps['cache']
        if is_prefix:
            model_ps  = copy.deepcopy(model_ps)
            model_ps.pop("cache")
        return pjit(_forward_generate,
                in_shardings=(model_ps, PS(), PS(), PS()),
                out_shardings=(PS(), PS(), cache_ps, PS())
                )
    if FLAGS.iterative_mode:
        prefix_forward_generate = create_forward_generate(model_ps, rpt_config.chunk_size, is_prefix=True)
        single_forward_generate = create_forward_generate(model_ps, rpt_config.chunk_size, is_prefix=False)
    else:
        create_forward_generate(model_ps, FLAGS.max_new_tokens, is_prefix=True)
    


    mesh = RPTConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        params = tree_apply(shard_fns, params)
        sharded_rng = next_rng()
    class ModelServer(LMServer):
        

        @staticmethod
        def loglikelihood_rolling(text):
            memory = Memory(chunk_size=64, num_neighbors=2, nearest_chunk_distance=FLAGS.nearest_chunk_distance)
            params.update(cache=jax.tree_map(lambda x:jnp.zeros_like(x) ,params['cache']))

            loglikelihood_list = []
            total_loglikelihood = 0.0
            total_is_greedy = True
            token_count = np.zeros((len(text),), dtype=np.int32)
            
            for batch in rolling_iterator(tokenizer, text, FLAGS.input_length):
                token_count+= batch['output_mask'].sum(-1)
                with mesh:
                    loglikelihood, is_greedy = forward_loglikelihood(
                        params, batch, memory
                    )
                    loglikelihood, is_greedy = jax.device_get((loglikelihood, is_greedy))
                total_loglikelihood += loglikelihood
                loglikelihood_list.append(loglikelihood.item())
                total_is_greedy = np.logical_and(is_greedy, total_is_greedy)
            print(loglikelihood_list)

            return total_loglikelihood, total_is_greedy, token_count
        
        @staticmethod
        def lowcoder_rolling(text,num_neighbors=2, wipe_cache=False,nearest_chunk_distance=None):
            if nearest_chunk_distance is None:
                nearest_chunk_distance = FLAGS.nearest_chunk_distance
            memory = Memory(chunk_size=64, num_neighbors=num_neighbors, nearest_chunk_distance=nearest_chunk_distance, is_dense=FLAGS.dense_mem)
            if wipe_cache:
                params.update(cache=jax.tree_map(lambda x:jnp.zeros_like(x) ,params['cache']))

            for batch in rolling_iterator(tokenizer, text, FLAGS.input_length):
                with mesh:
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
            params.update(cache=jax.tree_map(lambda x:jnp.zeros_like(x) ,params['cache']))
            batch = prefix_tokenizer(text,
                                     max_length=2*rpt_config.chunk_size,
                                     return_tensors='np',
                                     truncation=True,
                                     padding='max_length',)
            input_mask = batch.attention_mask.astype(int)
            batch = {"input_tokens":batch.input_ids.astype(int),
                    "input_mask":input_mask}
            with mesh:
                outputs, past_key_values = _forward_lowcoder(params, batch)
                params.update(cache=past_key_values)
            prompt_vector = outputs.encoded_hidden_states
            
            prompt_vector = prompt_vector.reshape([1,1,2*rpt_config.chunk_size,rpt_config.hidden_size])
            prompt_mask = input_mask.reshape([1,1,2*rpt_config.chunk_size])
            return prompt_vector, prompt_mask
        
        


        @staticmethod
        def generate(text, temperature, memory_str=None, prompt=None, max_new_tokens=64, precompile=False):
            n_turns = max(max_new_tokens//rpt_config.chunk_size, 1) if not precompile else 2
            if prompt is not None:
                prompt_vector, prompt_mask = ModelServer.lowcoder_single(prompt)
            if len(memory_str)>0:
                with open(memory_str) as f:
                    memory_text = f.read()
                _, _, memory = ModelServer.lowcoder_rolling(memory_text,num_neighbors=FLAGS.num_neighbors, nearest_chunk_distance=0)
            else:
                memory = Memory(chunk_size=64, num_neighbors=FLAGS.num_neighbors, nearest_chunk_distance=0, is_dense=FLAGS.dense_mem)
            
            nonlocal sharded_rng 
            inputs = prefix_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=FLAGS.input_length,
                return_tensors='np',
            )
            input_tokens = inputs.input_ids
            input_mask = inputs.attention_mask
            if FLAGS.add_bos_token:
                input_tokens[:, 0] = tokenizer.bos_token_id
                input_mask[:, 0] = 1
            batch = dict(
                input_tokens=input_tokens.astype(int),
                input_mask=input_mask.astype(int),
            )
            with mesh:
                outputs, past_key_values = _forward_lowcoder(params, batch)
                params.update(cache=past_key_values)
            neighbor_hidden_states, neighbor_mask, *_ = memory.add(
                                    input_tokens=batch["input_tokens"],
                                    encoded_hidden_states=outputs.encoded_hidden_states,
                                    key_chunks=outputs.key_chunks,
                                    query_chunks=outputs.query_chunks,
                                    append=False,
                                    )
            
            output_text = [tuple() for _ in range(input_tokens.shape[0])]
            
            params.pop("cache")
            output = None 
            for turn_index in range(n_turns+1):
                if turn_index==0: #first iteration
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
                        neighbor_hidden_states = np.concatenate([prompt_vector,neighbor_hidden_states],axis=1)
                        neighbor_mask = np.concatenate([prompt_mask,neighbor_mask],axis=1)
                    
                    with mesh:
                        neighbor_hidden_states, past_key_values = _forward_augment(
                            params, dict(hidden_states=enc_lowcoder_states.original_hidden_states,
                                        neighbor_hidden_states=neighbor_hidden_states,
                                        neighbor_mask=neighbor_mask)
                        )
                    params.update(cache=past_key_values)
                    
                    latest_token = output[:,-1:]
                    batch.update(input_tokens=latest_token,
                                input_mask=jnp.ones_like(latest_token, dtype=jnp.int32))
                    
                    chunk_index = jnp.zeros([1,1],dtype=jnp.int32) #we are assuming batch size =1 again..
                    forward_generate = single_forward_generate
                
                batch.update(
                    encoded_neighbors=EncodedNeighbors(
                                neighbor_hidden_states=neighbor_hidden_states, 
                                neighbor_mask=neighbor_mask,
                                chunk_index=chunk_index,
                                )
                )
                with mesh:
                    output, sharded_rng, past_key_values, enc_lowcoder_states = forward_generate(
                                params, sharded_rng, batch, temperature
                            )                    
                params.update(cache=past_key_values)
                output = jax.device_get(output)
                output_text = postproc_output(tokenizer, output, output_text, verbose=True)
                                

                
                
                
            return ["".join(x) for x in output_text]
        




    server = ModelServer(FLAGS.lm_server)
    server.run()


if __name__ == "__main__":
    mlxu.run(main)






        

