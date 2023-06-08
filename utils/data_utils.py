import numpy as np

# For torch Dataloader to collate jax numpy
def np_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [np_collate(samples) for samples in transposed]
  else:
    return np.array(batch)