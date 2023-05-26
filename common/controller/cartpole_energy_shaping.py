import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from common.dynamics.cartpole import Cartpole
from common.controller.controller import Controller

class CartpoleEnergyShapingController(Controller):
    def __init__(self, cartpole:Cartpole, Q=np.eye(4), R=np.eye(1), K=np.array([4, 4, 10]), eps_energy=1, eps_state=1) -> None:
        super().__init__()
        self.cartpole = cartpole
        self.xf = np.array([0, np.pi, 0, 0])
        self.umin, self.umax = self.cartpole.get_control_limit()

        # Hyperparameters for controllers
        self.Q = Q 
        self.R = R
        self.K = K
        self.eps_energy = eps_energy
        self.eps_state = eps_state

    def get_linearized_dynamics(self,):
        """
        Calculate the linearized dynamics around the terminal condition such that

        x_dot approximately equals to Alin @ x + Blin @ u

        returns:
            Alin: 4X4 matrix 
            Blin: 4X1 matrix 
        """

        M = self.cartpole.get_M(self.xf)
        B = self.cartpole.get_B()

        pGpq = np.array([[0,0], [0, -self.cartpole.mp * self.cartpole.g * self.cartpole.l]])

        Alin = np.vstack([np.array([[0, 0, 1, 0],[0, 0, 0, 1]]),
                        np.hstack([-np.linalg.inv(M)@pGpq, np.zeros((2,2))])
                        ])
        
        Blin = np.hstack([np.zeros(2),np.linalg.inv(M) @ B]).reshape(4,1)

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
    
    def get_control_efforts(self, x):
        """
        Calculate the control efforts applied to the acrobot
        
        params:
            x: current state, np.array of size (4,)
        returns:
            u: control effort applied to the robot, a scalar number
        """

        dx = self.cartpole.states_wrap((x-self.xf))

        K, P = self.get_lqr_term()

        de = self.energy(x) - self.energy(self.xf)

        if np.abs(de) < self.eps_energy and np.linalg.norm(dx[np.array([1,3])]) < self.eps_state:
            u = -K @ dx
        else:
            u = self.get_energy_shaping_input(x)

        u = np.clip(u, self.umin, self.umax)
        
        return u
    
    def energy(self, x):
        """
        Caculate a "energy" term for the pole, which is defined by 0.5 theta_dot ^2 - cos theta 
        """

        return 0.5 * x[3] **2 - np.cos(x[1])
    
    def get_energy_shaping_input(self, x):

        de = self.energy(x) - self.energy(self.xf)

        u_bar = de * x[3] * np.cos(x[1])

        ddq1_d = np.dot(self.K, np.array([-x[0], -x[2], u_bar]))

        ddq2_d = -np.cos(x[1]) / self.cartpole.l * ddq1_d - self.cartpole.g * np.sin(x[1]) / self.cartpole.l

        u = (self.cartpole.mc + self.cartpole.mp) * ddq1_d + self.cartpole.mp * self.cartpole.l * np.cos(x[1]) * ddq2_d \
            - self.cartpole.mp * self.cartpole.l * np.sin(x[1]) * x[3] **2

        return np.array([u])


def test_cartpole(cartpole: Cartpole, cartpole_controller: CartpoleEnergyShapingController):
    t0 = 0
    tf = 10
    t = np.arange(t0, tf, cartpole.dt)

    x0 = cartpole.get_initial_state()

    xs = [x0]
    us = []

    for i in range(1, t.shape[0]):
        us.append(cartpole_controller.get_control_efforts(xs[i-1]))
        xs.append(cartpole.simulate(xs[i-1],us[i-1]))
    
    energy = [cartpole_controller.energy(x) for x in xs]

    xs = np.array(xs)
    us = np.array(us)

    anim, fig = cartpole.plot_trajectory(t, xs)

    plt.figure(2); plt.clf()
    plt.plot(t[:-1], us)
    plt.xlabel("time(s)")
    plt.ylabel("input")
    plt.title("input vs time")

    plt.figure(3); 
    plt.plot(t, energy)
    plt.xlabel("times(s)")
    plt.ylabel("energy")
    plt.title("energy vs time")

    plt.figure(4);
    plt.plot(t, xs[:,0], label="position")
    plt.plot(t, xs[:,1], label="angle")
    plt.plot(t, xs[:,2], label="velocity")
    plt.plot(t, xs[:,3], label="angular velocity")
    plt.legend()
    plt.xlabel("times(s)")
    plt.ylabel("state")
    plt.title("states vs time")

    plt.show()
    plt.close()

if __name__ == "__main__":
    import os
    import gin
    from common.configs.dynamics.cartpole_config import CartpoleDynamicsConfig
    path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/cartpole.gin",
        )
    )
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = CartpoleDynamicsConfig()
    cartpole = Cartpole(dynamics_config)
    cartpole_controller = CartpoleEnergyShapingController(cartpole)
    test_cartpole(cartpole, cartpole_controller)