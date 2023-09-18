import os
if "DEBUG" in os.environ:
  import debugpy
  debugpy.listen(5678)
  print("Waiting for debugger attach")
  debugpy.wait_for_client()
  
import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
# from flax.training.train_state import TrainState
from mlxu.logging import prefix_metrics 
from EasyLM.train_state import TrainState
from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,write_parameter_info
)
from EasyLM.models.rpt.rpt_model import (
    RPTConfig, FlaxRPTForCausalLMModule,RetrieverSupervision
)
from jax.experimental import multihost_utils
import absl

absl.flags.DEFINE_multi_string(
'gin_file', None, 'List of paths to the config files.')
absl.flags.DEFINE_multi_string(
'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    eval_freq=1000,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)

import gin

def calculate_loss(output, batch):
    logits = output.logits
    loss, accuracy = cross_entropy_loss_and_accuracy(
        logits, batch['target_tokens'], batch['loss_masks']
    )
    metrics = {"loss": loss, "accuracy": accuracy}
    if output.retriever_output is not None:
        raw_aux_loss,valid_pairs = output.retriever_output.aux_loss
        aux_loss = raw_aux_loss.sum()/valid_pairs.sum()
        loss_scale = output.retriever_output.loss_scale
        if np.prod(loss_scale.shape)>1: #for now...
            loss_scale = loss_scale.mean()
        scaled_aux_loss = loss_scale*aux_loss
        metrics['aux_loss'] = aux_loss
        metrics['loss_scale'] = loss_scale
        metrics['scaled_aux_loss'] = scaled_aux_loss
        metrics['perplexity'] = jnp.exp(loss)
        if output.retriever_output.retrieval_metrics is not None:
            retrieval_metrics = jax.tree_map(lambda x:x.mean(),
                                            output.retriever_output.retrieval_metrics)
            metrics = {**metrics,**retrieval_metrics}
        
        l = loss+scaled_aux_loss
        print(l)
        return l, metrics
    else:
        return loss, metrics
    
def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    rpt_config = RPTConfig()
    jax.distributed.initialize()
    
    
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    variant.update(prefix_metrics(rpt_config.to_dict(),"rpt_config"))
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset,
                                          tokenizer=None,
                                          model_config=rpt_config)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset,
            tokenizer=None,
            model_config=rpt_config
        )
        eval_iterator = iter(eval_dataset)
    

    seq_length = dataset.seq_length


    

    
    if rpt_config.vocab_size < dataset.vocab_size:
        print(f"overriding vocab_size={dataset.vocab_size}")
        
        rpt_config.update(dict(vocab_size=dataset.vocab_size))
    print(f"rpt_config: {rpt_config}")
    model = FlaxRPTForCausalLMModule(
        rpt_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )
    has_cca = model.config.cca_freq is not None

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(RPTConfig.get_weight_decay_exclusions())
    )

    device_count = jax.device_count()
    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        init_dict = dict(
            input_ids=jnp.zeros((device_count, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((device_count, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((device_count, seq_length), dtype=jnp.int32),
            rngs=rng_generator(rpt_config.rng_keys()),)
        if has_cca:
            num_total_chunk_scores = model.config.num_sequence_chunks*model.config.num_scored_neighbors
            init_dict['retriever_supervision'] = RetrieverSupervision(
                nei_scores = jnp.zeros((device_count,num_total_chunk_scores ), dtype=jnp.float32),
                nei_idx = jnp.zeros((device_count, num_total_chunk_scores), dtype=jnp.int32)
            )
        output, params = model.init_with_output(**init_dict)
        init_dict['target_tokens'] = jnp.zeros((device_count, seq_length), dtype=jnp.int32)
        init_dict['loss_masks'] = jnp.ones((device_count, seq_length), dtype=jnp.int32)
        _, metrics = calculate_loss(output, init_dict)
        
        return TrainState.create(params=params,
                                 tx=optimizer,
                                 metric_names=list(metrics.keys()),
                                 apply_fn=None)
        
    def create_trainstate_from_params(params, rng):
        state = init_fn(rng)
        return state.replace(params=params)
        

    def train_step(train_state, rng, batch):
        deterministic=False
        
        rng_generator = JaxRNG(rng)
        rngs = rng_generator(rpt_config.rng_keys())
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        if has_cca:
            model_input = {"input_ids": batch['input_tokens'],
                           "attention_mask": batch['attention_mask'],
                           "retriever_supervision": batch['retriever_supervision'],
                           "train_step":train_state.step}
        else:
            model_input = {"input_ids": batch['input_tokens'],}
            
        def loss_and_accuracy(params):
            output = model.apply(
                params,
                deterministic=deterministic,
                rngs=rngs,
                **model_input
            )
            return calculate_loss(output, batch)
        
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (_, metrics), grads = grad_fn(train_state.params)
        loss = metrics['loss']
        train_state,metrics = train_state.apply_gradients(grads=grads,metrics=metrics)
        
        
        metrics['learning_rate'] = optimizer_info['learning_rate_schedule'](train_state.step)
        metrics['step'] = train_state.step-1
        metrics["raw_loss"] = loss
        
        return train_state, rng_generator(), prefix_metrics(metrics, "train")

    def eval_step(train_state, rng, batch):
        deterministic=True
        rng_generator = JaxRNG(rng)
        rngs = rng_generator(rpt_config.rng_keys())
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        if has_cca:
            model_input = {"input_ids": batch['input_tokens'],
                           "attention_mask": batch['attention_mask'],
                           "retriever_supervision": batch['retriever_supervision'],
                           "train_step":train_state.step}
        else:
            model_input = {"input_ids": batch['input_tokens'],}
            
        output = model.apply(
                train_state.params,
                deterministic=deterministic,
                rngs=rngs,
                **model_input
            )
        logits = output.logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = {"loss": loss, "accuracy": accuracy,"step":train_state.step}
        if output.retriever_output is not None:
            raw_aux_loss,valid_pairs = output.retriever_output.aux_loss
            aux_loss = raw_aux_loss.sum()/valid_pairs.sum()
            
            loss_scale = output.retriever_output.loss_scale
            if np.prod(loss_scale.shape)>1: #for now...
                loss_scale = loss_scale.mean()
            scaled_aux_loss = loss_scale*aux_loss
            metrics['aux_loss'] = aux_loss
            metrics['loss_scale'] = loss_scale
            metrics['scaled_aux_loss'] = scaled_aux_loss
            metrics['perplexity'] = jnp.exp(loss)
            if output.retriever_output.retrieval_metrics is not None:
                retrieval_metrics = jax.tree_map(lambda x:x.mean(),
                                                 output.retriever_output.retrieval_metrics)
                metrics = {**metrics,**retrieval_metrics}
        return rng_generator(), prefix_metrics(metrics, "eval")

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        RPTConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, PS() ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,        
        in_shardings=(train_state_partition, PS(), PS(("dp","fsdp"))),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS(("dp","fsdp"))),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            rpt_config=rpt_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = RPTConfig.get_jax_mesh(FLAGS.mesh_dim)
    
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params, next_rng())
            del restored_params
        write_parameter_info(train_state)

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()
            

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            # if step == 100:
                # break
            log_metrics = {**dataset_metrics,
                           "steps_per_second":step_counter.format_dict['rate']}
            batch = multihost_utils.host_local_array_to_global_array(batch,
                                            mesh, 
                                            pspecs=PS((('dp', 'fsdp'))))
            with jax.profiler.StepTraceAnnotation("step", step_num=step):
                train_state, sharded_rng, metrics = sharded_train_step(
                    train_state, sharded_rng, batch
                )
            log_metrics.update(metrics)
            
            if step % FLAGS.eval_freq == 0:
                if int(FLAGS.eval_steps) > 0:
                    tqdm.write("\n" +f"performing {int(FLAGS.eval_steps)} eval steps" + "\n")
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        eval_batch = multihost_utils.host_local_array_to_global_array(eval_batch,
                                mesh, 
                                pspecs=PS((('dp', 'fsdp'))))

                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(jax.device_get(eval_metrics))
                    eval_metric_list = average_metrics(eval_metric_list)
                    
                    log_metrics.update(eval_metric_list)
            if step % FLAGS.log_freq == 0:
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
            
            logger.log(log_metrics)
                
                
                

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    import jax.profiler
    jax.profiler.start_server(9999)
    # with jax.profiler.trace("/tmp/tensorboard"):
    mlxu.run(main)
