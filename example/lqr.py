import numpy as np
import matplotlib.pyplot as plt
import gin
import os
import jax
from typing import Tuple
from common.dynamics.linear import LinearDynamics
from common.controller.lqr import LQR
from common.controller.vhjb import VHJBController
from common.configs.dynamics.linear_config import LinearDynamicsConfig
from common.configs.controller.vhjb_controller_config import VHJBControllerConfig

def load_config():
    path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../common/configs/dynamics/linear.gin",
        )
    )
    path_to_controller_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../common/configs/controller/linear_vhjb_controller.gin",
        )
    )
    
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = LinearDynamicsConfig()

    gin.parse_config_file(path_to_controller_config_file)
    controller_config = VHJBControllerConfig()

    return dynamics_config, controller_config

def test_policy(nn_policy: VHJBController, dynamics: LinearDynamics, lqr_controller:LQR, T=5):
    nn_policy.train_mode = False

    t_span = np.arange(0, T, dynamics.dt)
    xs_lqr = np.zeros((t_span.shape[0], dynamics.state_dim))
    us_lqr = np.zeros((t_span.shape[0]-1, dynamics.control_dim))
    xs_learned = np.zeros_like(xs_lqr)
    us_learned = np.zeros_like(us_lqr)
    cost_lqr = np.zeros(t_span.shape[0]-1)
    cost_learned = np.zeros(t_span.shape[0]-1)

    x0 = dynamics.get_initial_state()
    xs_lqr[0] = x0
    xs_learned[0] = x0
    for i in range(1, t_span.shape[0]):
        u_learned, v_gradient, updated_states = nn_policy.get_control_efforts(nn_policy.model_params, nn_policy.model_states, xs_learned[i-1])
        us_learned[i-1] = np.asarray(u_learned)
        us_lqr[i-1] = lqr_controller.get_control_efforts(xs_lqr[i-1])
        
        xs_learned[i] = dynamics.simulate(xs_learned[i-1], us_learned[i-1])
        xs_lqr[i] = dynamics.simulate(xs_lqr[i-1], us_lqr[i-1])

        cost_lqr[i-1] = nn_policy.running_cost(xs_lqr[i-1], us_lqr[i-1])
        cost_learned[i-1] = nn_policy.running_cost(xs_learned[i-1], us_learned[i-1])

    plt.figure()
    for i in range(xs_learned.shape[1]):
        plt.plot(t_span, xs_learned[:,i], label=f"learned x[{i}]")
        plt.plot(t_span, xs_lqr[:,i], label=f"lqr x[{i}]")
    plt.title("states vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure()
    for i in range(us_learned.shape[1]):
        plt.plot(t_span[:-1], us_learned[:,i], label=f"learned u[{i}]")
        plt.plot(t_span[:-1], us_lqr[:,i], label=f"lqr u[{i}]")
    plt.title("input vs time")
    plt.xlabel("time")
    plt.ylabel("input")
    plt.legend()

    plt.figure()
    plt.plot(t_span[:-1], np.cumsum(cost_learned), label="learned")
    plt.plot(t_span[:-1], np.cumsum(cost_lqr), label="lqr")
    plt.xlabel("time")
    plt.ylabel("cumulated cost")
    plt.title("cumulated cost vs time")
    plt.legend()

def draw_Value_contour(nn_policy: VHJBController, lqr_controller: LQR, x_mean: np.ndarray, indices: Tuple[int, int], x_range: np.ndarray, num_of_step=50):
    quick_apply = jax.jit(nn_policy.value_function_approximator.apply, static_argnames=["train"])

    x1 = np.linspace(-x_range[indices[0]], x_range[indices[0]], num_of_step)
    x2 = np.linspace(-x_range[indices[1]], x_range[indices[1]], num_of_step)
    X1, X2 = np.meshgrid(x1, x2)
    v_learned = np.zeros_like(X1)
    v_lqr = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X2.shape[1]):
            x = np.copy(x_mean)
            x[indices[0]] += X1[i,j]
            x[indices[1]] += X2[i,j]
            v_learned[i,j] = quick_apply({"params":nn_policy.model_params, **nn_policy.model_states}, x, train=False)
            v_lqr[i,j] = x.T @ lqr_controller.P @ x

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, v_lqr)
    ax.set_xlabel(f'x{indices[0]}')
    ax.set_ylabel(f'x{indices[1]}')
    ax.set_zlabel('value')
    plt.title("LQR value function levelset")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, v_learned)
    ax.set_xlabel(f'x{indices[0]}')
    ax.set_ylabel(f'x{indices[1]}')
    ax.set_zlabel('value')
    plt.title("Learned value function levelset")

def local_optimal_x(x: np.ndarray, nn_policy: VHJBController, max_iter=10, lr=1e-1, verbose=True, newton_method=True):
    """
    This function aim to find a local minimal value around x,
    the local minimal can further served for debug purpose. 
    """
    nn_policy.train_mode = False

    quick_hess_fn = jax.jit(jax.hessian(nn_policy.value_function_approximator.apply, argnums=1), static_argnames=["train"])
    
    for i in range(max_iter):
        value = nn_policy.value_function_approximator.apply({"params":nn_policy.model_params, **nn_policy.model_states}, x, train=False)
        u, v_gradient, updated_states = nn_policy.get_control_efforts(nn_policy.model_params, nn_policy.model_states, x)
        hess = quick_hess_fn({"params":nn_policy.model_params, **nn_policy.model_states}, x, train=False)
        if verbose:
            print(f"iter:{i}, x: {x}, value:{value:.5f}, \n u:{u} v_gradient:{v_gradient} \n hess:{hess}")
        if np.linalg.det(hess) > 0 and newton_method:
            x -= lr * jax.numpy.linalg.inv(hess) @ v_gradient
        else:
            x -= lr * v_gradient
    return x

def main():
    dynamics_config, controller_config = load_config()    
    dynamics = LinearDynamics(dynamics_config)
    nn_policy = VHJBController(dynamics, controller_config)
    lqr_controller = LQR(dynamics, np.asarray(controller_config.Q), np.asarray(controller_config.R))
    nn_policy.train()
    test_policy(nn_policy, dynamics, lqr_controller)
    draw_Value_contour(nn_policy, lqr_controller, x_mean=np.array([0.0,0,0,0]), indices=(0,1), x_range=np.array([2.4, 0.3, 5, 5]))
    plt.show()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()