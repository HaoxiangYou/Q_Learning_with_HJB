import numpy as np
from typing import Tuple
from  scipy.integrate import solve_ivp
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
        np.random.seed(config.seed)

    def get_initial_state(self,) -> np.ndarray:
        return np.random.uniform(low=-self.x0_std, high=self.x0_std) + self.x0_mean

    def get_dimension(self,) -> Tuple[int, int]:
        """
            returns:
                states dim, control dim
        """
        return self.state_dim, self.control_dim

    def get_control_affine_matrix(self, x) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_control_limit(self,) -> Tuple[np.ndarray, np.ndarray]:
        return self.umin, self.umax
    
    def get_M(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_C(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_G(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_B(self) -> np.ndarray:
        raise NotImplementedError
    
    def states_wrap(self, x:np.ndarray) -> np.ndarray:
        """
        wrap the states into valid range

        params:
            x: state vector in (states_dim, )
        returns:
            wrap states in (states_dim, )
        """
        raise NotImplementedError
    
    def get_control_affine_matrix(self, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        f_1 = np.zeros((state_dim,))
        f_2 = np.zeros((state_dim,control_dim))


        q, dq = x[:(state_dim//2)], x[(state_dim//2):]

        M = self.get_M(x)

        C = self.get_C(x)

        G = self.get_G(x)
        B = self.get_B()

        f_1[:(state_dim//2)] = dq
        f_1[(state_dim//2):] = -np.linalg.inv(M) @ (np.dot(C, dq)  + G)

        f_2[(state_dim//2):,:] = (np.linalg.inv(M) @ B).reshape(-1,control_dim) 

        return f_1, f_2
    
    def dynamics_step(self, x, u):
        """
            Caclulate x_dot for single x and u
        """

        f_1, f_2 = self.get_control_affine_matrix(x)

        xdot = f_1 + f_2 @ u

        return xdot.squeeze()

    def simulate(self, x, u, dt=None):
        """
        Simulate the open loop acrobot for one step
        """

        # make sure u is within the range
        u = np.clip(u, self.umin, self.umax)
        
        def f(t, x):
            return self.dynamics_step(x,u)
        if dt:
            sol = solve_ivp(f, (0,dt), x)
        else:
            sol = solve_ivp(f, (0,self.dt), x)

        state = self.states_wrap(sol.y[:,-1].ravel()).squeeze()

        return state