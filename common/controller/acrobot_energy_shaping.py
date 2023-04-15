import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from common.dynamics.acrobot import Acrobot, p, dt

def wrap(q):
    return (q + np.pi) %(2*np.pi) - np.pi

class AcrobotEnergyShapingController():
    """
    Acrobot controller implement with energy shaping methods
    """
    def __init__(self, acrobot_system: Acrobot, Q=np.eye(4), R=np.eye(1), eps=1000, K=np.array([1, 2, 1])):
        self.acrobot = acrobot_system 
        self.xf = np.array([np.pi,0,0,0])

        # Hyperparameters for controllers
        self.K = K
        self.Q = Q
        self.R = R
        self.eps = eps

    def get_linearized_dynamics(self):
        """
        Calculate the linearized dynamics around the terminal condition such that

        x_dot approximately equals to Alin @ x + Blin @ u

        returns:
            Alin: 4X4 matrix 
            Blin: 4X1 matrix 
        """

        M = self.acrobot.get_M(self.xf)
        B = self.acrobot.get_B()

        M_inv = np.linalg.inv(M)

        pGpq1 = np.array([-self.acrobot.m1*self.acrobot.g*self.acrobot.l1/2  -self.acrobot.m2*self.acrobot.g*self.acrobot.l1 - self.acrobot.m2*self.acrobot.g*self.acrobot.l2/2, 
                        -self.acrobot.m2*self.acrobot.g*self.acrobot.l2/2])
        
        pGpq2 = np.array([-self.acrobot.m2*self.acrobot.g*self.acrobot.l2/2, -self.acrobot.m2*self.acrobot.g*self.acrobot.l2/2])

        Alin = np.vstack(
            [np.array([0,0,1,0]), 
            np.array([0,0,0,1]),
            np.hstack([-np.linalg.inv(M)@pGpq1.reshape(self.acrobot.dim,1), -np.linalg.inv(M)@pGpq2.reshape(self.acrobot.dim,1), np.zeros((2,2))])])

        Blin = np.hstack([np.zeros(self.acrobot.dim),M_inv @ B]).reshape(self.acrobot.dim*2,1)

        return Alin, Blin

    def get_lqr_term(self):
        """
        Calculate the lqr terms for linearized dynamics:

        The control input and cost to go can be calculated as:
            u(dx) = -K dx
            V(dx) = dx^T P @ dx

        returns:
            K: 1X4 matrix
            P: 4x4 matrix
        """

        Alin, Blin = self.get_linearized_dynamics()

        P = sp.linalg.solve_continuous_are(Alin, Blin, self.Q, self.R)
        K = np.dot(sp.linalg.inv(self.R), np.dot(Blin.T, P))

        return K, P
    
    # Spong's paper, energy based swingup
    def get_swingup_input(self, x):
        """
        Calculate the swingup input using energy shaping method

        params:
            x: curret state, (4,) np.arrary
        
        returns:
            u: the input applied to the robot, a scalar number
        """
        
        q, dq = x[:self.acrobot.dim], x[self.acrobot.dim:]

        M = self.acrobot.get_M(x)
        C = self.acrobot.get_C(x)
        
        G = self.acrobot.get_G(x)

        ubar = (self.acrobot.energy(x) - self.acrobot.energy(self.xf))*dq[0]

        ddq2_d = np.dot(self.K, np.array([-wrap(q[1]), -dq[1], ubar]))

        u = (M[1,1] - M[0,1]**2/M[0,0])*ddq2_d + (G + C @ dq)[1] - M[1,0]/M[0,0] * (G + C @ dq)[0]

        return np.array([u])
    
    def get_control_efforts(self, x):
        """
        Calculate the control efforts applied to the acrobot
        
        params:
            x: current state, np.array of size (4,)
        returns:
            u: control effort applied to the robot, a scalar number
        """

        dx = np.hstack([wrap((x - self.xf)[:self.acrobot.dim]), (x - self.xf)[self.acrobot.dim:]])

        K, P = self.get_lqr_term()

        if dx.T @ P @ dx< self.eps:
            u = -K @ dx
        else:
            u = self.get_swingup_input(x)

        u = np.clip(u, -self.acrobot.umax, self.acrobot.umax)
        
        return u

def test_acrobot(acrobot:Acrobot, acrobot_controller:AcrobotEnergyShapingController):

    t0 = 0
    tf = 25
    t = np.arange(t0, tf, dt)

    xs = np.zeros((t.shape[0], 4))
    us = np.zeros((t.shape[0]-1,))
    xs[0] = np.array([0.001, 0, 0, 0])
    
    for i in range(1, xs.shape[0]):
        us[i-1] = acrobot_controller.get_control_efforts(xs[i-1])
        xs[i] = acrobot.simulate(xs[i-1],us[i-1],dt)

    xs[:,0] = np.arctan2(np.sin(xs[:,0]), np.cos(xs[:,0]))
    xs[:,1] = np.arctan2(np.sin(xs[:,1]), np.cos(xs[:,1]))
    e = np.array([acrobot.energy(x) for x in xs])

    knee = p['l1']*np.cos(xs[:,0]-np.pi/2), p['l1']*np.sin(xs[:,0]-np.pi/2)
    toe = p['l1']*np.cos(xs[:,0]-np.pi/2) + p['l2']*np.cos(xs[:,0]+xs[:,1]-np.pi/2), \
        p['l1']*np.sin(xs[:,0]-np.pi/2) + p['l2']*np.sin(xs[:,0]+xs[:,1]-np.pi/2)

    anim, fig = acrobot.plot_trajectory(t, xs)

    plt.figure(2); plt.clf()
    plt.plot(knee[0], knee[1], 'k.-', lw=0.5, label='knee')
    plt.plot(toe[0], toe[1], 'b.-', lw=0.5, label='toe')
    plt.xlabel('x'); plt.ylabel('y')
    plt.plot(np.linspace(-2,2,100),
             (p['l1']+p['l2'])*np.ones(100), 'r', lw=1,
             label='max height')
    plt.title("position")
    plt.legend()

    plt.figure(3); plt.clf();
    plt.plot(t, e, 'k.-', lw=0.5)
    plt.axhline(acrobot.energy(np.array([np.pi, 0, 0, 0])), linestyle='--', color="r")
    plt.title('E')

    plt.show()
    plt.close()

    return xs, e

if __name__ == "__main__":
    acrobot = Acrobot()
    acrobot_controller = AcrobotEnergyShapingController(acrobot)
    test_acrobot(acrobot, acrobot_controller)