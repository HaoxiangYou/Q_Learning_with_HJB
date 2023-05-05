from dataclasses import dataclass
import numpy as np
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
        self.x0_mean = np.array(self.x0_mean)
        self.x0_std = np.array(self.x0_std)
        self.umin = np.array(self.umin)
        self.umax = np.array(self.umax)
        self.state_dim = self.x0_mean.shape[0]
        self.control_dim = self.umin.shape[0]