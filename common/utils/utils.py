import jax.numpy as jnp

# For torch Dataloader to collate jax numpy
def jnp_collate(batch):
  if isinstance(batch[0], jnp.ndarray):
    return jnp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [jnp_collate(samples) for samples in transposed]
  else:
    return jnp.array(batch)