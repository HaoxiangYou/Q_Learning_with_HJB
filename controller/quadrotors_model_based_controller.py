import numpy as np
import scipy.linalg
from controller.controller_basic import Controller
from dynamics.quadrotors import Quadrotors2D
from typing import List, Tuple

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
    
class QuadrotorsWaypointsPlanner():

    def __init__(self, waypoints: np.ndarray, dynamics: Quadrotors2D,  avg_speed=0.25) -> None:
        if waypoints.shape[1] == 2:
            self.dim = 2
        elif waypoints.shape[1] == 3:
            self.dim = 3
        else:
            raise ValueError("The waypoints dim should either be 2D or 3D")        

        self.points = waypoints
        self.dynamics = dynamics
        self.points_num = waypoints.shape[0]
        self.avg_speed = avg_speed
        self.displacements = self.points[1:] - self.points[:-1]
        self.distants = (np.sum(self.displacements **2, axis=1))**0.5
        # Currently the time of each segment is assigned by distance / avg_speed 
        self.interval_t = self.distants / self.avg_speed
        self.cumulated_t = np.zeros(self.points_num)
        self.cumulated_t[1:] = np.cumsum(self.interval_t)
        self.coeff = self.solve_minimum_snap_coefficient()

    def solve_minimum_snap_coefficient(self) -> np.ndarray:
        """
        Solving a minimum snap trajectory coefficient for quadrotors
        The minimum snap trajectory is a 7th order polynominal,

        Returns:
            coefficients: as dim X points_num-1 X 8 numpy array
        """

        A = np.zeros(((self.points_num-1)*8, (self.points_num-1)*8))
        b = np.zeros((self.dim, (self.points_num-1)*8))
        coeff= np.zeros((self.dim, self.points_num-1, 8))

        # End points constraints
        # t[0] at p[0], t[-1] at p[-1]
        A[0, :8] = self.get_polynomial_term(t=0, n=0, order=7)
        A[1, -8:] = self.get_polynomial_term(t=self.interval_t[-1], n=0, order=7)
        for i in range(self.dim):
            b[i][0] = self.points[0][i]
            b[i][1] = self.points[-1][i]
        # v, acc, jerk is zero at begin and end
        A[2, :8] = self.get_polynomial_term(t=0, n=1, order=7)
        A[3, -8:] = self.get_polynomial_term(t=self.interval_t[-1], n=1, order=7)
        A[4, :8] = self.get_polynomial_term(t=0, n=2, order=7)
        A[5, -8:] = self.get_polynomial_term(t=self.interval_t[-1], n=2, order=7)
        A[6, :8] = self.get_polynomial_term(t=0, n=3, order=7)
        A[7, -8:] = self.get_polynomial_term(t=self.interval_t[-1], n=3, order=7)

        # Intermediate points position constraints
        for i in range(self.points_num - 2):
            A[8+i*2, i*8:(i+1)*8] = self.get_polynomial_term(t=self.interval_t[i], n=0)
            A[9+i*2, (i+1)*8:(i+2)*8] = self.get_polynomial_term(t=0, n=0, order=7)
            for j in range(self.dim):
                b[j][8+i*2] = self.points[i+1][j]
                b[j][9+i*2] = self.points[i+1][j]
        
        # Intermediate points continuous constraints:
        for i in range(self.points_num - 2):
            # Velocity continuous
            A[8 + (self.points_num-2)*2 + i*6, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=1, order=7)
            A[8 + (self.points_num-2)*2 + i*6, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=1, order=7)
            # Acceleration continuous
            A[8 + (self.points_num-2)*2 + i*6 + 1, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=2, order=7)
            A[8 + (self.points_num-2)*2 + i*6 + 1, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=2, order=7)
            # Jerk continuous
            A[8 + (self.points_num-2)*2 + i*6 + 2, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=3, order=7)
            A[8 + (self.points_num-2)*2 + i*6 + 2, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=3, order=7)
            # Snap continunous
            A[8 + (self.points_num-2)*2 + i*6 + 3, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=4, order=7)
            A[8 + (self.points_num-2)*2 + i*6 + 3, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=4, order=7)
            # Crackle continunous
            A[8 + (self.points_num-2)*2 + i*6 + 4, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=5, order=7)
            A[8 + (self.points_num-2)*2 + i*6 + 4, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=5, order=7)
            # 6th order continunous
            A[8 + (self.points_num-2)*2 + i*6 + 5, 8*i:8*(i+1)] = self.get_polynomial_term(t=self.interval_t[i], n=6, order=7)
            A[8 + (self.points_num-2)*2 + i*6 + 5, 8*(i+1):8*(i+2)] = - self.get_polynomial_term(t=0, n=6, order=7)

        for i in range(self.dim):
            coeff[i] = np.array(np.linalg.solve(A, b[i])).reshape(self.points_num-1,8)
        
        return coeff
 
    def get_polynomial_term(self, t, n, order=7) -> np.ndarray:
        """
        Given polynomial, it's nth derivative f^{(n)}(t), can be represent as c^T X z(t), where c is the coefficent of the polynomial,

        Args:
            t: time since the begin of current segment
            n: nth derivative
            order: the order of polynomial
        Returns:
            a numpy array to multiply the coefficient  
        """            
        z = np.zeros(order+1)
        for i in range(order+1):
            if i-n >= 0:
                z[i] = t **(i-n) * np.prod(np.arange(i, i-n, -1))
        return z

    def flat_output_to_full_states_and_inputs_2D(self, t:np.ndarray, coeff:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate full states and corresponding inputs for quadrotors 2D given the flat output.
        Args:
            t: the time since the begin of current segment
            coeff: coefficients for x and y polynominal
        Returns:
            full_states: [x, y, theta, x_dot, y_dot, theta_dot]
            inputs: [u_1, u_2]
        """
        order = coeff.shape[1] - 1
        x = np.dot(coeff[0], self.get_polynomial_term(t=t, n=0, order=order))
        y = np.dot(coeff[1], self.get_polynomial_term(t=t, n=0, order=order))
        x_dot = np.dot(coeff[0], self.get_polynomial_term(t=t, n=1, order=order))
        y_dot = np.dot(coeff[1], self.get_polynomial_term(t=t, n=1, order=order))
        x_ddot = np.dot(coeff[0], self.get_polynomial_term(t=t, n=2, order=order))
        y_ddot = np.dot(coeff[1], self.get_polynomial_term(t=t, n=2, order=order))
        x_dddot = np.dot(coeff[0], self.get_polynomial_term(t=t, n=3, order=order))
        y_dddot = np.dot(coeff[1], self.get_polynomial_term(t=t, n=3, order=order))
        x_ddddot = np.dot(coeff[0], self.get_polynomial_term(t=t, n=4, order=order))
        y_ddddot = np.dot(coeff[1], self.get_polynomial_term(t=t, n=4, order=order))

        theta = -np.arctan2(x_ddot , y_ddot + self.dynamics.g)
        theta_dot = -1 / (1+ (x_ddot/ (y_ddot + self.dynamics.g))**2) * (x_dddot / (y_ddot + self.dynamics.g) - x_ddot * y_dddot / (y_ddot + self.dynamics.g)**2)
        theta_ddot = 2 * x_ddot/ (y_ddot + self.dynamics.g) / (1+ (x_ddot/ (y_ddot + self.dynamics.g))**2) **2 * (x_dddot / (y_ddot + self.dynamics.g) - x_ddot * y_dddot / (y_ddot + self.dynamics.g)**2) **2 \
                    - 1 / (1+ (x_ddot/ (y_ddot + self.dynamics.g))**2) * (x_ddddot / (y_ddot + self.dynamics.g) - x_dddot * y_dddot / (y_ddot + self.dynamics.g)**2 - x_dddot *  y_dddot / (y_ddot + self.dynamics.g)**2 - x_ddot * y_ddddot / (y_ddot + self.dynamics.g)**2 + 2 * x_ddot * y_dddot / (y_ddot + self.dynamics.g) **3)

        full_state = np.array([x,y,theta,x_dot,y_dot,theta_dot])

        if not np.sin(theta) == 0:
            u_1 = (self.dynamics.I / self.dynamics.r * theta_ddot - self.dynamics.m / np.sin(theta) * x_ddot) / 2
            u_2 = (-self.dynamics.I / self.dynamics.r * theta_ddot- self.dynamics.m / np.sin(theta) * x_ddot) / 2
        else:
            u_1 = (self.dynamics.I / self.dynamics.r * theta_ddot + self.dynamics.m / np.cos(theta) * (y_ddot + self.dynamics.g)) / 2
            u_2 = (-self.dynamics.I / self.dynamics.r * theta_ddot + self.dynamics.m / np.cos(theta) * (y_ddot + self.dynamics.g)) / 2

        u = np.array([u_1, u_2])

        return full_state, u

    def update(self, t):
        index = np.argwhere(t>=self.cumulated_t)[-1,0]
        if index == self.cumulated_t.shape[0] - 1:
            # if the time is larger than last segment, then hovering
            t_diff = 0
            coeff = np.zeros((self.dim, self.coeff.shape[1]))
            for i in range(self.dim):
                coeff[i][0] = self.points[-1][i]
        else:
            t_diff = t-self.cumulated_t[index]
            coeff = self.coeff[:,index, :]
        
        if self.dim == 2:
            return self.flat_output_to_full_states_and_inputs_2D(t_diff, coeff)
        else:
            raise NotImplementedError
        
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