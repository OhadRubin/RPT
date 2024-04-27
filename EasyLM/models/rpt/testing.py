import jax.numpy as jnp
from jax import lax


x = jnp.zeros((10, ))
y = jnp.ones((7, ))
print(lax.dynamic_update_slice(x, y, (5, )))