import numpy as np
from common.dynamics.dynamics import Dynamics
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

dt = 0.05

p = {   'l1':0.5, 'l2':1,
        'm1':8, 'm2':8,
        'I1':2, 'I2':8,
        'umax': 25,
        'g': 10,
        'dt':dt,
    }

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
        self.g, self.dt = self.p['g'], self.p['dt']

    def get_dimension(self) -> Tuple[int, int]:
        return self.dim*2, 1
    
    def get_control_limit(self) -> Tuple[np.ndarray, np.ndarray]:
        return -np.array([self.umax]), np.array([self.umax])

    def get_M(self, x: np.ndarray) -> np.ndarray:
        M = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.l2/2*np.cos(x[1]),
                            self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1])],
                    [self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1]), self.I2]])
        return M 
    
    def get_C(self, x: np.ndarray) -> np.ndarray:
        q, dq = x[:self.dim], x[self.dim:]
        C = np.array([[-2*self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1], -self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1]],
                    [self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[0], 0]])
        return C

    def get_G(self, x: np.ndarray) -> np.ndarray:
        G = np.array([(self.m1*self.l1/2 + self.m2*self.l1)*self.g*np.sin(x[0]) + self.m2*self.g*self.l2/2*np.sin(x[0]+x[1]),
                        self.m2*self.g*self.l2/2*np.sin(x[0]+x[1])])
        return G
    
    def get_B(self) -> np.ndarray:
        B = np.array([0, 1])
        return B

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
    
    def states_wrap(self, x:np.ndarray) -> np.ndarray:
        assert x.shape == (4,)
        x[:2] = np.remainder(x[:2] + np.pi, 2*np.pi) - np.pi
        return x
    
    def plot_trajectory(self, ts:np.ndarray, xs:np.ndarray, margin=0.5):
        fig = plt.figure()
        ax = plt.axes()

        knees = self.l1*np.cos(xs[:,0]-np.pi/2), self.l1*np.sin(xs[:,0]-np.pi/2)
        toes = self.l1*np.cos(xs[:,0]-np.pi/2) + self.l2*np.cos(xs[:,0]+xs[:,1]-np.pi/2), \
            self.l1*np.sin(xs[:,0]-np.pi/2) + self.l2*np.sin(xs[:,0]+xs[:,1]-np.pi/2)


        def draw_frame(i):
            ax.clear()
            ax.set_xlim(-(self.l1+self.l2+margin), (self.l1+self.l2+margin))
            ax.set_ylim(-(self.l1+self.l2+margin), (self.l1+self.l2+margin))
            ax.axhline(y=0, color='r', linestyle= '-')
            ax.plot(0,0,'bo', label="joint")
            ax.plot(knees[0][i], knees[1][i], 'bo')
            ax.plot(knees[0][i]/2, knees[1][i]/2, 'go', label="center of mass")
            ax.plot([0, knees[0][i]], [0, knees[1][i]], 'k-', label="linkage")
            ax.plot(toes[0][i], toes[1][i], 'bo')
            ax.plot((toes[0][i] + knees[0][i])/2, (toes[1][i] + knees[1][i])/2, 'go')
            ax.plot([knees[0][i],toes[0][i]], [knees[1][i], toes[1][i]], 'k-')
            ax.legend()
            ax.set_title("{:.1f}s".format(ts[i]))
        
        anim = animation.FuncAnimation(fig, draw_frame, frames=ts.shape[0], repeat=False, interval=self.dt*1000)

        return anim, fig