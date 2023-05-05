from dataclasses import dataclass
import gin
import numpy as np
from typing import Sequence 
from common.configs.dynamics.dynamics_config import DynamicsConfig

@gin.configurable
@dataclass
class CartpoleDynamicsConfig(DynamicsConfig):
    mc: float
    mp: float
    g: float
    l: float

    def __post_init__(self):
        super().__post_init__()
        