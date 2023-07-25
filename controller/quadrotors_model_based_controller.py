import numpy as np
import scipy.linalg
from controller.controller_basic import Controller
from dynamics.quadrotors import Quadrotors2D

class Quadrotors2DHoveringController(Controller):
    """
        This controller trying to stablize and hover 2D quadrotors at given place.
    """
    def __init__(self, dynamics:Quadrotors2D, xf:np.ndarray, Q:np.ndarray, R:np.ndarray) -> None:
        super().__init__()
        
        self.dynamics = dynamics
        self.xf = xf
        self.Q = Q
        self.R = R
        self.umin, self.umax = self.dynamics.get_control_limit()
        
        if np.linalg.norm(xf[2:]) > 0:
            raise ValueError("Final Velocity or Angle is not zero")
        
        self.uf = self.dynamics.m*self.dynamics.g/2 * np.ones(2)

        self.A = np.vstack([np.hstack([np.zeros((3,3)), np.eye(3)]),
                np.array([0,0,-self.dynamics.g, 0,0,0]),
                np.zeros((2,6))
                ])
        self.B = np.vstack([np.zeros((4,2)),
                            np.ones((1,2)) / self.dynamics.m,
                            np.array([self.dynamics.r/self.dynamics.I, -self.dynamics.r/self.dynamics.I])])
        
        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.dot(scipy.linalg.inv(self.R), np.dot(self.B.T, self.P))

    def get_control_efforts(self, x:np.ndarray) -> np.ndarray:
        
        return np.clip(-self.K @ self.dynamics.states_wrap(x - self.xf) + self.uf, self.umin, self.umax)
    
if __name__ == "__main__":
    import os
    import gin
    import matplotlib.pyplot as plt
    from configs.dynamics.dynamics_config import Quadrotors2DConfig
    path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/quadrotors2D.gin",
        )
    )
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = Quadrotors2DConfig()
    quadrotors2D = Quadrotors2D(dynamics_config)
    quadrotors2DHoveringController =  Quadrotors2DHoveringController(quadrotors2D, quadrotors2D.x0_mean, np.eye(6), np.eye(2))

    t0 = 0
    tf = 5
    t = np.arange(t0, tf, quadrotors2D.dt)
    x0 = quadrotors2D.get_initial_state()

    xs = [x0]
    us = []

    for i in range(1, t.shape[0]):
        us.append(quadrotors2DHoveringController.get_control_efforts(xs[i-1]))
        xs.append(quadrotors2D.simulate(xs[i-1],us[i-1]))

    xs = np.array(xs)
    us = np.array(us)

    anim, fig = quadrotors2D.plot_trajectory(t, xs)

    plt.figure()
    for i in range(xs.shape[1]):
        plt.plot(t, xs[:,i], label=f"learned x[{i}]")
    plt.title("states vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure()
    for i in range(us.shape[1]):
        plt.plot(t[:-1], us[:,i], label=f"learned u[{i}]")
    plt.title("input vs time")
    plt.xlabel("time")
    plt.ylabel("input")
    plt.legend()
    
    plt.show()
    plt.close()

    import pdb; pdb.set_trace()