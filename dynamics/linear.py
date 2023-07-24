import numpy as np
import jax.numpy as jnp
from dynamics.dynamics_basic import Dynamics
from configs.dynamics.dynamics_config import LinearDynamicsConfig
from typing import Tuple, Union

class LinearDynamics(Dynamics):
    def __init__(self, config: LinearDynamicsConfig) -> None:
        super().__init__(config)
        assert config.A.ndim == 2
        assert config.B.ndim == 2
        assert config.A.shape[0] == config.B.shape[0]
        assert config.A.shape[0] == config.A.shape[1] 
        self.A = config.A
        self.B = config.B

    def states_wrap(self, x:Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        return x
    
    def get_control_affine_matrix(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert x.shape[0] == self.state_dim
        return self.A @ x, self.B