from dataclasses import dataclass
import jax.numpy as jnp
from typing import Sequence 

@dataclass
class DynamicsConfig:
    seed: int
    dt: float
    umin: Sequence[float]
    umax: Sequence[float]
    x0_mean: Sequence[float]
    x0_std: Sequence[float]

    def __post_init__(self):
        self.x0_mean = jnp.array(self.x0_mean, dtype=jnp.float32)
        self.x0_std = jnp.array(self.x0_std, dtype=jnp.float32)
        self.umin = jnp.array(self.umin, dtype=jnp.float32)
        self.umax = jnp.array(self.umax, dtype=jnp.float32)
        self.state_dim = self.x0_mean.shape[0]
        self.control_dim = self.umin.shape[0]