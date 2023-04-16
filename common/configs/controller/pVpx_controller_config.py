from dataclasses import dataclass
import gin
import numpy as np
from typing import Sequence 

@gin.configurable
@dataclass
class pVpxControllerConfig:
    # general
    seed: int
    
    # neural network hyperparameters
    hidden_size: int
    lr: float
    device: str
    epochs: int
    warm_up_epochs: int
    batch_size: int
    
    # initial states
    x0_mean: Sequence[float]
    x0_std: Sequence[float]
    num_of_warmup_trajectory: int
    
    # states sample
    xs_mean: Sequence[float]
    xs_std: Sequence[float]
    sample_size: int

    # system asscociate
    Q: Sequence[Sequence[float]]
    R: Sequence[Sequence[float]]
    xf: Sequence[float]

    def __post_init__(self):
        self.x0_mean = np.array(self.x0_mean)
        self.x0_std = np.array(self.x0_std)
        self.xs_mean = np.array(self.xs_mean)
        self.xs_std = np.array(self.xs_std)
        self.Q = np.array(self.Q)
        self.R = np.array(self.R)
        self.xs = np.array(self.xf)
