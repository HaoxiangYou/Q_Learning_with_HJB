import argparse
import gin
import os
import numpy as np
import matplotlib.pyplot as plt
from dynamics.linear import LinearDynamics
from controller.lqr import LQR
from controller.vhjb import VHJBController
from configs.dynamics.linear_config import LinearDynamicsConfig
from configs.controller.vhjb_controller_config import VHJBControllerConfig
from utils.debug_plots import visualize_loss_landscope, visualize_value_landscope
from functools import partial

def load_config():
    path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/linear.gin",
        )
    )
    path_to_controller_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/controller/linear_vhjb_controller.gin",
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
        us_learned[i-1] = nn_policy.get_control_efforts(xs_learned[i-1])
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

def main():
    dynamics_config, controller_config = load_config()    
    dynamics = LinearDynamics(dynamics_config)
    nn_policy = VHJBController(dynamics, controller_config)
    lqr_controller = LQR(dynamics, np.asarray(controller_config.Q), np.asarray(controller_config.R))

    nn_policy.train()
    
    test_policy(nn_policy, dynamics, lqr_controller)
    
    visualize_value_landscope(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, lambda x: x.T @ lqr_controller.P @ x,
                              x_mean=controller_config.normalization_mean, indices=(0,1), x_range=controller_config.normalization_std)

    xs, costs, dones = next(iter(nn_policy.dataloader))
    visualize_loss_landscope(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, nn_policy.key,
                            xs, partial(nn_policy.hjb_loss), dones=dones)
    visualize_loss_landscope(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, nn_policy.key,
                            xs, partial(nn_policy.termination_loss), dones=dones, costs=costs)
    plt.show()
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()