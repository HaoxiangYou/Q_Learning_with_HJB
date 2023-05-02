import numpy as np
import torch
from common.dynamics.dynamics import Dynamics
from common.configs.dynamics.linear_config import LinearDynamicsConfig
from typing import Tuple

class LinearDynamics(Dynamics):
    def __init__(self, config: LinearDynamicsConfig) -> None:
        super().__init__()
        assert config.A.ndim == 2
        assert config.B.ndim == 2
        assert config.A.shape[0] == config.B.shape[0]
        assert config.A.shape[0] == config.A.shape[1] 

        self.A = config.A
        self.B = config.B

        self.umin = config.umin
        self.umax = config.umax

        self.dt = config.dt

    def get_dimension(self):
        return self.A.shape[1], self.B.shape[1]
    
    def get_control_limit(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.umin, self.umax

    def states_wrap(self, x:np.ndarray) -> np.ndarray:
        return x
    
    def get_control_affine_matrix(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state_dim, control_dim = self.get_dimension()
        assert x.shape[0] == state_dim
        return self.A @ x, self.B