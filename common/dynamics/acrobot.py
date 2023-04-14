import numpy as np
import torch
from scipy.integrate import solve_ivp
from common.dynamics.dynamics import Dynamics
from typing import Tuple

dt = 0.05
x0 = np.array([0.001, 0, 0, 0])
xf = np.array([np.pi,0,0,0])

p = {   'l1':0.5, 'l2':1,
        'm1':8, 'm2':8,
        'I1':2, 'I2':8,
        'umax': 25,
        'g': 10,
        'dt':dt,
        'xf':xf,
    }

def wrap(q):
    return (q + np.pi) %(2*np.pi) - np.pi

# From Russ Tedrake's notes
class Acrobot(Dynamics):

    def __init__(self, params=p):
        super().__init__()
        self.dim = 2
        self.control_dim = 1

        # Define params for the system
        self.p = params
        self.g = 10
        self.m1, self.m2, self.l1, self.l2, self.I1, self.I2, self.umax = \
            self.p['m1'], self.p['m2'], self.p['l1'], self.p['l2'], self.p['I1'], self.p['I2'], self.p['umax']
        self.g, self.xf, self.dt = self.p['g'], self.p['xf'], self.p['dt']

    def get_dimension(self):
        return self.dim*2, 1

    def get_M(self, x):
        M = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.l2/2*np.cos(x[1]),
                            self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1])],
                    [self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1]), self.I2]])
        return M 
    
    def get_C(self, x):
        q, dq = x[:self.dim], x[self.dim:]
        C = np.array([[-2*self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1], -self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1]],
                    [self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[0], 0]])
        return C

    def get_G(self, x):
        G = np.array([(self.m1*self.l1/2 + self.m2*self.l1)*self.g*np.sin(x[0]) + self.m2*self.g*self.l2/2*np.sin(x[0]+x[1]),
                        self.m2*self.g*self.l2/2*np.sin(x[0]+x[1])])
        return G
    
    def get_B(self,):
        B = np.array([0, 1])
        return B

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

        f_1 = np.zeros((n, self.dim*2))
        f_2 = np.zeros((n, self.dim*2,self.control_dim))

        for i in range(n):
            x = xs[i]

            q, dq = x[:self.dim], x[self.dim:]

            M = self.get_M(x)

            C = self.get_C(x)

            G = self.get_G(x)
            B = self.get_B()

            f_1[i,:self.dim] = dq
            f_1[i,self.dim:] = -np.linalg.inv(M) @ (np.dot(C, dq)  + G)

            f_2[i, self.dim:,:] = (np.linalg.inv(M) @ B).reshape(-1,self.control_dim) 

        return f_1, f_2

    def get_control_limit(self) -> Tuple[np.ndarray, np.ndarray]:
        return -np.array([self.umax]), np.array([self.umax])

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
        u = np.clip(u, -self.umax, self.umax)
        
        def f(t, x):
            return self.dynamics_step(x,u)
        sol = solve_ivp(f, (0,dt), x, first_step=dt)
        return sol.y[:,-1].ravel()

    def energy(self, x):
        q, dq = x[:self.dim], x[self.dim:]
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])

        T1 = 0.5*self.I1*dq[0]**2
        T2 = 0.5*(self.m2*self.l1**2 + self.I2 + 2*self.m2*self.l1*self.l2/2*c2)*dq[0]**2 \
            + 0.5*self.I2*dq[1]**2 \
            + (self.I2 + self.m2*self.l1*self.l2/2*c2)*dq[0]*dq[1]
        U  = -self.m1*self.g*self.l1/2*c1 - self.m2*self.g*(self.l1*c1 + self.l2/2*np.cos(q[0]+q[1]))
        return T1 + T2 + U