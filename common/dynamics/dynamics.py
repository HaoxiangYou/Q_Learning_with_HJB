import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union
from common.configs.dynamics.dynamics_config import DynamicsConfig

class Dynamics:

    dt: float
    state_dim: int
    control_dim: int
    x0_mean: np.ndarray
    x0_std: np.ndarray
    umin: np.ndarray
    umax: np.ndarray

    def __init__(self, config: DynamicsConfig) -> None:
        self.state_dim = config.state_dim
        self.control_dim = config.control_dim
        self.dt = config.dt
        self.umin = config.umin
        self.umax = config.umax
        self.x0_mean = config.x0_mean
        self.x0_std = config.x0_std
        self.random_key = jax.random.PRNGKey(config.seed)
        np.random.seed(config.seed)

    def get_initial_state(self,) -> np.ndarray:
        return self.states_wrap(np.random.uniform(size=(self.state_dim, ), low=-self.x0_std, high=self.x0_std) + self.x0_mean)

    def get_dimension(self,) -> Tuple[int, int]:
        """
            returns:
                states dim, control dim
        """
        return self.state_dim, self.control_dim
    
    def get_control_limit(self,) -> Tuple[np.ndarray, np.ndarray]:
        return self.umin, self.umax
    
    def get_M(self, x:Union[jnp.ndarray, np.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_C(self, x:Union[jnp.ndarray, np.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_G(self, x:Union[jnp.ndarray, np.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_B(self) -> np.ndarray:
        raise NotImplementedError
    
    def states_wrap(self, x:Union[np.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        """
        wrap the states into valid range

        params:
            x: state vector in (states_dim, )
        returns:
            wrap states in (states_dim, )
        """
        raise NotImplementedError
    
    def get_control_affine_matrix(self, x:Union[jnp.ndarray, np.ndarray]) \
        -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
            x_dot = f_1(x) + f_2(x) u
            params:
                x: state vector in (states_dim, )
            return:
                f_1(x): 
                    in (state_dim, )
                f_2(x):
                    in (state_dim, control_dim)
        """
        assert x.ndim == 1

        q, dq = x[:(self.state_dim//2)], x[(self.state_dim//2):]

        M = self.get_M(x)

        C = self.get_C(x)

        G = self.get_G(x)
        B = self.get_B()

        if isinstance(x, jnp.ndarray):
            f_1 = jnp.hstack([dq,-jnp.linalg.inv(M) @ (jnp.dot(C, dq)  + G)])
            f_2 = jnp.vstack([jnp.zeros((self.state_dim//2, self.control_dim)), (jnp.linalg.inv(M) @ B).reshape(-1,self.control_dim)]) 
        else:
            f_1 = np.hstack([dq,-np.linalg.inv(M) @ (np.dot(C, dq)  + G)])
            f_2 = np.vstack([np.zeros((self.state_dim//2, self.control_dim)), (np.linalg.inv(M) @ B).reshape(-1,self.control_dim)]) 

        return f_1, f_2
    
    def dynamics_step(self, x:Union[jnp.ndarray, np.ndarray], u:Union[jnp.ndarray, np.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        """
            Caclulate x_dot for single x and u
        """

        f_1, f_2 = self.get_control_affine_matrix(x)

        xdot = f_1 + f_2 @ u

        return xdot.squeeze()

    def simulate(self, x:np.ndarray, u:np.ndarray):
        """
        Simulate the open loop acrobot for one step
        """

        # Type check, jnp array was on purpose not allowed, 
        # because initial a small jnp array is costly 
        if not (isinstance(x, np.ndarray), isinstance(u, np.ndarray)):
            raise TypeError("The x and u should all be np.ndarray")

        # make sure u is within the range
        u = np.clip(u, self.umin, self.umax)

        state = self.states_wrap(x + self.dynamics_step(x, u)*self.dt)

        return state