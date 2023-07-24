import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union
from configs.dynamics.dynamics_config import Quadrotors2DConfig
from dynamics.dynamics_basic import Dynamics
import matplotlib.pyplot as plt
from matplotlib import animation

class Quadrotors2D(Dynamics):
    def __init__(self, config: Quadrotors2DConfig) -> None:
        super().__init__(config)
        self.g = config.g
        self.m = config.m
        self.r = config.r
        self.I = config.I

    def get_control_affine_matrix(self, x:Union[jnp.ndarray, np.ndarray]) \
        -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
            x_dot = f_1(x) + f_2(x) u
            
            args:
                x: state vector in (states_dim, )
            returns:
                f_1(x): 
                    in (state_dim, )
                f_2(x):
                    in (state_dim, control_dim)
        """
        assert x.ndim == 1
        q, dq = x[:(self.state_dim//2)], x[(self.state_dim//2):]
        if isinstance(x, jnp.ndarray):
            f_1 = jnp.hstack([dq, jnp.array([0, -self.g, 0])])
            f_2 = jnp.vstack([jnp.zeros((self.state_dim//2, self.control_dim)),
                            -jnp.sin(q[2])/self.m * jnp.ones((1, self.control_dim)),
                            jnp.cos(q[2])/self.m * jnp.ones((1, self.control_dim)),
                            jnp.array([self.r/self.I, -self.r/self.I])
                            ])
        else:
            f_1 = np.hstack([dq, np.array([0, -self.g, 0])])
            f_2 = np.vstack([np.zeros((self.state_dim//2, self.control_dim)),
                            -np.sin(q[2])/self.m * np.ones((1, self.control_dim)),
                            np.cos(q[2])/self.m * np.ones((1, self.control_dim)),
                            np.array([self.r/self.I, -self.r/self.I])
                            ])
        return f_1, f_2
    
    def states_wrap(self, x:Union[np.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, np.ndarray]:
        """
            Wrap the orientation to [-pi, pi). 
            Will also need to turn the trajectory orientation to this range if tracking nominant trajectory,
            another choice may leave orientation alone for both generated trajectory and dynamics
            
            args:
                x: state vector in (states_dim, ) or (batch_size, state_dim)
            returns:
                wrap states in (states_dim, ) or (batch_size, state_dim)
        """
        assert x.shape == (6,) or (x.shape[1] == 6 and x.ndim == 2)
        if isinstance(x, jnp.ndarray):
            if x.ndim == 2:
                return x.at[:,2].set(jnp.remainder(x[:,2] + jnp.pi, 2*jnp.pi) - jnp.pi)
            else:
                return x.at[2].set(jnp.remainder(x[2] + jnp.pi, 2*jnp.pi) - jnp.pi)
        else:
            if x.ndim == 2:
                x[:,2] = np.remainder(x[:,2] + np.pi, 2*np.pi) - np.pi 
            else:
                x[2] = np.remainder(x[2] + np.pi, 2*np.pi) - np.pi 
            return x
        
    def plot_trajectory(self, ts:np.ndarray, xs:np.ndarray):
        
        fig = plt.figure()
        ax = plt.axes()

        x_min = np.min(xs[:, 0])
        x_max = np.max(xs[:, 0])
        y_min = np.min(xs[:, 1])
        y_max = np.max(xs[:, 1])

        def draw_frame(i):            
            ax.clear()
            ax.axis('equal')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            ax.plot(xs[:i+1, 0], xs[:i+1, 1], label="actual trajectory", color="g")

            ax.plot([xs[i, 0] - self.r * np.cos(xs[i, 2]), xs[i, 0] + self.r * np.cos(xs[i, 2])],
                    [xs[i, 1] - self.r * np.sin(xs[i, 2]), xs[i, 1] + self.r * np.sin(xs[i, 2])],
                    color="b", label="quadrotors", linewidth=3)
            
            ax.legend()
            ax.set_title("{:.1f}s".format(ts[i]))

        anim = animation.FuncAnimation(fig, draw_frame, frames=ts.shape[0], repeat=False, interval=self.dt*1000)

        return anim, fig

if __name__ == "__main__":
    import os
    import gin
    path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/quadrotors2D.gin",
        )
    )
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = Quadrotors2DConfig()
    quadrotors2D = Quadrotors2D(dynamics_config)
    t0 = 0
    tf = 1
    t = np.arange(t0, tf, quadrotors2D.dt)
    x0 = quadrotors2D.get_initial_state()

    xs = [x0]
    us = []

    for i in range(1, t.shape[0]):
        us.append(np.zeros(2))
        xs.append(quadrotors2D.simulate(xs[i-1],us[i-1]))

    xs = np.array(xs)
    us = np.array(us)

    anim, fig = quadrotors2D.plot_trajectory(t, xs)

    plt.show()
    plt.close()
