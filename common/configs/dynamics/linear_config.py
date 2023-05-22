from dataclasses import dataclass
import gin
import jax.numpy as jnp
from typing import Sequence
from common.configs.dynamics.dynamics_config import DynamicsConfig

@gin.configurable
@dataclass
class LinearDynamicsConfig(DynamicsConfig):
    A: Sequence[Sequence[float]]
    B: Sequence[Sequence[float]]

    def __post_init__(self):
        super().__post_init__()
        self.A = jnp.array(self.A, dtype=jnp.float32)
        self.B = jnp.array(self.B, dtype=jnp.float32)