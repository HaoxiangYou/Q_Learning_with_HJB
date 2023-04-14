import numpy as np
import torch
from typing import Tuple
from  scipy.integrate import solve_ivp
class Dynamics:
    def __init__(self,) -> None:
        pass
    
    def get_dimension(self,) -> Tuple[int, int]:
        """
            returns:
                states dim, control dim
        """
        raise NotImplementedError

    def get_control_affine_matrix(self, x) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_control_limit(self,) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_M(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_C(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_G(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_B(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_control_affine_matrix(self, xs) -> Tuple[np.ndarray, np.ndarray]:
        """
            x_dot = f_1(x) + f_2(x) u
            params:
                xs: n X self.dim*2
            return:
                f_1(x): 
                    n X self.dim*2  tensor or array
                f_2(x):
                    n X self.dim*2 X self.control_dim tensor or array
        """
        # If x is a single vector turn it into a 1 X self.dim*2
        if xs.ndim == 1:
            xs = xs.reshape(1,-1)
        # If x is a tensor turn it into arrary
        if isinstance(xs, torch.Tensor):
            xs = xs.numpy()

        n = xs.shape[0]

        state_dim, control_dim = self.get_dimension()

        f_1 = np.zeros((n, state_dim))
        f_2 = np.zeros((n, state_dim,control_dim))

        for i in range(n):
            x = xs[i]

            q, dq = x[:(state_dim//2)], x[(state_dim//2):]

            M = self.get_M(x)

            C = self.get_C(x)

            G = self.get_G(x)
            B = self.get_B()

            f_1[i,:(state_dim//2)] = dq
            f_1[i,(state_dim//2):] = -np.linalg.inv(M) @ (np.dot(C, dq)  + G)

            f_2[i, (state_dim//2):,:] = (np.linalg.inv(M) @ B).reshape(-1,control_dim) 

        return f_1, f_2
    
    def dynamics_step(self, x, u):
        """
            Caclulate x_dot for single x and u
        """

        f_1, f_2 = self.get_control_affine_matrix(x)

        # Turn f_1, f_2 into single vector and u into a scalar

        if isinstance(u, torch.Tensor):
            u = u.item()
        if isinstance(u, np.ndarray):
            u = u.squeeze()

        f_1 = f_1.squeeze()
        f_2 = f_2.squeeze()

        xdot = f_1 + f_2 * u

        return xdot

    def simulate(self, x, u, dt):
        """
        Simulate the open loop acrobot for one step
        """

        # make sure u is within the range
        umin, umax = self.get_control_limit()
        u = np.clip(u, umin, umax)
        
        def f(t, x):
            return self.dynamics_step(x,u)
        sol = solve_ivp(f, (0,dt), x, first_step=dt)
        return sol.y[:,-1].ravel()