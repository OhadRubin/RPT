# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable
from flax import core
from flax import struct
import optax



import jax
import math
from flax import struct
import jax.numpy as jnp
# "gradient_norm","param_norm"
METRIC_NAMES = ["loss","accuracy","aux_loss","scaled_aux_loss","loss_scale","perplexity"] + [f"recall@{i}" for i in [2,5,10,20]]
class RollingAverage(struct.PyTreeNode):
  size: int 
  last_element: int
  mat: jnp.ndarray

  def update(self, new_value):
    mat = self.mat.at[self.last_element].set(new_value)
    last_element = (self.last_element+1) % mat.shape[0]
    size = jnp.where(self.size!=mat.shape[0],self.size+1,self.size)
    curr_value = mat.sum()/size

    return curr_value,self.replace(size=size,
                        last_element=last_element,
                        mat=mat,
                        )
  @classmethod
  def create(cls, *, size):
    return cls(
        size=0,
        last_element=0,
        mat=jnp.zeros(size,dtype=jnp.float32)
    )

class RollingAverageTree(struct.PyTreeNode):
  roll_avg_obj: struct.field(pytree_node=True) 
  def update(self, new_values):
    curr_value_dict = {}
    curr_roll_avg_obj = {}
    for name,value in new_values.items():
      curr_value,curr_obj = self.roll_avg_obj[name].update(value)
      curr_roll_avg_obj[name] = curr_obj
      curr_value_dict[name] = curr_value
    return curr_value_dict,self.replace(roll_avg_obj=curr_roll_avg_obj)
  
  @classmethod
  def create(cls, *, size,names):
    return cls(
        roll_avg_obj={name:RollingAverage.create(size=size) for name in names}
    )

class TrainState(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.

  Synopsis::

      state = TrainState.create(
          apply_fn=model.apply,
          params=variables['params'],
          tx=tx)
      grad_fn = jax.grad(make_loss_fn(state.apply_fn))
      for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """

  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  metric_state:struct.PyTreeNode = struct.field(pytree_node=True)

  def apply_gradients(self, *, grads,metrics, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    values, metric_state = self.metric_state.update(metrics)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        metric_state=metric_state,
        **kwargs,
    ),values

  @classmethod
  def create(cls, *, apply_fn, params, tx, metric_names=METRIC_NAMES,metric_buffer=25, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    metric_state = RollingAverageTree.create(size=metric_buffer,names = metric_names)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        metric_state = metric_state,
        **kwargs,
    )