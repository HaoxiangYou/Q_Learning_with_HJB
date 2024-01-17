from dataclasses import dataclass
import numpy as np
import gin
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
        self.x0_mean = np.array(self.x0_mean, dtype=np.float32)
        self.x0_std = np.array(self.x0_std, dtype=np.float32)
        self.umin = np.array(self.umin, dtype=np.float32)
        self.umax = np.array(self.umax, dtype=np.float32)
        self.state_dim = self.x0_mean.shape[0]
        self.control_dim = self.umin.shape[0]

@gin.configurable
@dataclass
class LinearDynamicsConfig(DynamicsConfig):
    A: Sequence[Sequence[float]]
    B: Sequence[Sequence[float]]

    def __post_init__(self):
        super().__post_init__()
        self.A = np.array(self.A, dtype=np.float32)
        self.B = np.array(self.B, dtype=np.float32)

@gin.configurable
@dataclass
class CartpoleDynamicsConfig(DynamicsConfig):
    mc: float
    mp: float
    g: float
    l: float

    def __post_init__(self):
        super().__post_init__()

@gin.configurable
@dataclass
class Quadrotors2DConfig(DynamicsConfig):
    g: float
    m: float
    r: float
    I: float

@gin.configurable
@dataclass
class NearHoverQuadcopterConfig(DynamicsConfig):
    g: float
    m: float
    kT: float
    n0: float