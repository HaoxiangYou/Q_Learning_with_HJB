from dataclasses import dataclass
import gin
import numpy as np
from typing import Sequence 

@gin.configurable
@dataclass
class LinearDynamicsConfig:
    A: Sequence[Sequence[float]]
    B: Sequence[Sequence[float]]
    dt: float
    umin: Sequence[float]
    umax: Sequence[float]

    def __post_init__(self):
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        self.umin = np.array(self.umin)
        self.umax = np.array(self.umax)