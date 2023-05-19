from dataclasses import dataclass
import gin
import jax.numpy as jnp
from typing import Sequence

@gin.configurable
@dataclass
class VHJBControllerConfig:
    # general
    seed: int

    # neural model hyperparameters
    features: Sequence[int]
    normalization_mean: Sequence[float]
    normalization_std: Sequence[float]
    
    # training hyperparameters
    lr: float
    warmup_epochs: int
    epochs: int
    batch_size: int
    regularization: float

    # trajectories sample hyperparameters
    warmup_trajectories: int
    num_of_trajectories_per_epoch: int
    maximum_step: int
    input_std: float
    std_decay_rate: float
    std_decay_step: int
    maximum_buffer_size: int

    # task related hyperparameters
    Q: Sequence[Sequence[float]]
    R: Sequence[Sequence[float]]
    xf: Sequence[float]
    obs_min: Sequence[float]
    obs_max: Sequence[float]

    def __post_init__(self):
        self.normalization_mean = jnp.array(self.normalization_mean, dtype=jnp.float32)
        self.normalization_std = jnp.array(self.normalization_std, dtype=jnp.float32)
        self.Q = jnp.array(self.Q, dtype=jnp.float32)
        self.R = jnp.array(self.R, dtype=jnp.float32)
        self.xf = jnp.array(self.xf, dtype=jnp.float32)
        self.obs_min = jnp.array(self.obs_min, dtype=jnp.float32)
        self.obs_max = jnp.array(self.obs_max, dtype=jnp.float32)
