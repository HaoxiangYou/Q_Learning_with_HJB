from dataclasses import dataclass
import gin
import numpy as np
from typing import Sequence
from configs.dynamics.dynamics_config import DynamicsConfig

@gin.configurable
@dataclass
class LinearDynamicsConfig(DynamicsConfig):
    A: Sequence[Sequence[float]]
    B: Sequence[Sequence[float]]

    def __post_init__(self):
        super().__post_init__()
        self.A = np.array(self.A, dtype=np.float32)
        self.B = np.array(self.B, dtype=np.float32)