import jax
import jax.numpy as jnp
from typing import Tuple
from common.configs.dynamics.dynamics_config import DynamicsConfig
class Dynamics:

    dt: float
    state_dim: int
    control_dim: int
    x0_mean: jnp.ndarray
    x0_std: jnp.ndarray
    umin: jnp.ndarray
    umax: jnp.ndarray

    def __init__(self, config: DynamicsConfig) -> None:
        self.state_dim = config.state_dim
        self.control_dim = config.control_dim
        self.dt = config.dt
        self.umin = config.umin
        self.umax = config.umax
        self.x0_mean = config.x0_mean
        self.x0_std = config.x0_std
        self.random_key = jax.random.PRNGKey(config.seed)

    def get_initial_state(self,) -> jnp.ndarray:
        self.random_key, key_to_use = jax.random.split(self.random_key)
        return self.states_wrap(jax.random.uniform(key_to_use, shape=(self.state_dim, ), minval=-self.x0_std, maxval=self.x0_std) + self.x0_mean)

    def get_dimension(self,) -> Tuple[int, int]:
        """
            returns:
                states dim, control dim
        """
        return self.state_dim, self.control_dim

    def get_control_affine_matrix(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def get_control_limit(self,) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.umin, self.umax
    
    def get_M(self, x:jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    def get_C(self, x:jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    def get_G(self, x:jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    def get_B(self) -> jnp.ndarray:
        raise NotImplementedError
    
    def states_wrap(self, x:jnp.ndarray) -> jnp.ndarray:
        """
        wrap the states into valid range

        params:
            x: state vector in (states_dim, )
        returns:
            wrap states in (states_dim, )
        """
        raise NotImplementedError
    
    def get_control_affine_matrix(self, x:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

        state_dim, control_dim = self.get_dimension()

        q, dq = x[:(state_dim//2)], x[(state_dim//2):]

        M = self.get_M(x)

        C = self.get_C(x)

        G = self.get_G(x)
        B = self.get_B()

        f_1 = jnp.hstack([dq,-jnp.linalg.inv(M) @ (jnp.dot(C, dq)  + G)])

        f_2 = jnp.vstack([jnp.zeros((state_dim//2, control_dim)), (jnp.linalg.inv(M) @ B).reshape(-1,control_dim)]) 

        return f_1, f_2
    
    def dynamics_step(self, x, u):
        """
            Caclulate x_dot for single x and u
        """

        f_1, f_2 = self.get_control_affine_matrix(x)

        xdot = f_1 + f_2 @ u

        return xdot.squeeze()

    def simulate(self, x, u):
        """
        Simulate the open loop acrobot for one step
        """

        # make sure u is within the range
        u = jnp.clip(u, self.umin, self.umax)

        state = self.states_wrap(x + self.dynamics_step(x, u)*self.dt)

        return state