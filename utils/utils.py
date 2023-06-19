import numpy as np
from functools import partial

# For torch Dataloader to collate numpy
def np_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def keep_first_element(func):
    """
    Decorator that modifies the return value of a function
    to keep only the first element if it's a tuple.
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        else:
            return result
    return wrapper


