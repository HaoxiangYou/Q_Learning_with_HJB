import argparse
import gin
import os
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from dynamics.dynamics import Dynamics
from dynamics.linear import LinearDynamics
from dynamics.cartpole import Cartpole
from controller.controller import Controller
from controller.vhjb import VHJBController
from controller.cartpole_energy_shaping import CartpoleEnergyShapingController
from controller.lqr import LQR
from configs.dynamics.dynamics_config import LinearDynamicsConfig, CartpoleDynamicsConfig
from configs.controller.vhjb_controller_config import VHJBControllerConfig
from utils.debug_plots import visualize_loss_landscope, visualize_value_landscope_for_lqr

def load_systems(env_name, dynamics_config, vhjb_controller_config):
    if env_name == 'lqr':
        dynamics, nn_policy, model_based_policy = load_lqr(dynamics_config, vhjb_controller_config)
    elif env_name == 'cartpole':
        dynamics, nn_policy, model_based_policy = load_cartpole(dynamics_config, vhjb_controller_config)

    return dynamics, nn_policy, model_based_policy
    
def load_lqr(dynamics_config, vhjb_controller_config):
    if dynamics_config is None:
        path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/linear.gin",
            )
        )
    else:
        path_to_dynamics_config_file = dynamics_config

    if vhjb_controller_config is None:
        path_to_vhjb_controller_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/controller/linear_vhjb_controller.gin",
            )
        )
    else:
        path_to_vhjb_controller_config_file = vhjb_controller_config
    
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = LinearDynamicsConfig()

    gin.parse_config_file(path_to_vhjb_controller_config_file)
    vhjb_controller_config = VHJBControllerConfig()

    dynamics = LinearDynamics(dynamics_config)
    nn_policy = VHJBController(dynamics, vhjb_controller_config)
    # initial lqr with same Q and R as neural network controller
    lqr_controller = LQR(dynamics, np.asarray(vhjb_controller_config.Q), np.asarray(vhjb_controller_config.R))

    return dynamics, nn_policy, lqr_controller

def load_cartpole(dynamics_config, vhjb_controller_config):
    if dynamics_config is None:
        path_to_dynamics_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/dynamics/cartpole.gin",
            )
        )
    else:
        path_to_dynamics_config_file = dynamics_config

    if vhjb_controller_config is None:
        path_to_vhjb_controller_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/controller/cartpole_vhjb_controller.gin",
            )
        )
    else:
        path_to_vhjb_controller_config_file = vhjb_controller_config
    
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = CartpoleDynamicsConfig()

    gin.parse_config_file(path_to_vhjb_controller_config_file)
    vhjb_controller_config = VHJBControllerConfig()

    dynamics = Cartpole(dynamics_config)
    nn_policy = VHJBController(dynamics, vhjb_controller_config)
    # initial lqr with same Q and R as neural network controller
    energy_shaping_controller = CartpoleEnergyShapingController(dynamics, np.asarray(vhjb_controller_config.Q), np.asarray(vhjb_controller_config.R))

    return dynamics, nn_policy, energy_shaping_controller

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

def test_policy(nn_policy: VHJBController, dynamics: Dynamics, model_based_controller:Controller, T=5):
    nn_policy.train_mode = False

    t_span = np.arange(0, T, dynamics.dt)
    xs_model_based = np.zeros((t_span.shape[0], dynamics.state_dim))
    us_model_based = np.zeros((t_span.shape[0]-1, dynamics.control_dim))
    xs_learned = np.zeros_like(xs_model_based)
    us_learned = np.zeros_like(us_model_based)
    cost_lqr = np.zeros(t_span.shape[0]-1)
    cost_learned = np.zeros(t_span.shape[0]-1)

    x0 = dynamics.get_initial_state()
    xs_model_based[0] = x0
    xs_learned[0] = x0
    for i in range(1, t_span.shape[0]):
        us_learned[i-1] = nn_policy.get_control_efforts(xs_learned[i-1])
        us_model_based[i-1] = model_based_controller.get_control_efforts(xs_model_based[i-1])
        
        xs_learned[i] = dynamics.simulate(xs_learned[i-1], us_learned[i-1])
        xs_model_based[i] = dynamics.simulate(xs_model_based[i-1], us_model_based[i-1])

        cost_lqr[i-1] = nn_policy.running_cost(xs_model_based[i-1], us_model_based[i-1]) * dynamics.dt
        cost_learned[i-1] = nn_policy.running_cost(xs_learned[i-1], us_learned[i-1]) * dynamics.dt


    anim, fig = None, None
    if hasattr(dynamics, 'plot_trajectory'):
        anim, fig = dynamics.plot_trajectory(t_span, xs_learned)

    plt.figure()
    for i in range(xs_learned.shape[1]):
        plt.plot(t_span, xs_learned[:,i], label=f"learned x[{i}]")
        plt.plot(t_span, xs_model_based[:,i], label=f"lqr x[{i}]")
    plt.title("states vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure()
    for i in range(us_learned.shape[1]):
        plt.plot(t_span[:-1], us_learned[:,i], label=f"learned u[{i}]")
        plt.plot(t_span[:-1], us_model_based[:,i], label=f"lqr u[{i}]")
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

    return anim, fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='lqr', choices=['lqr', 'cartpole'], help="Environment name")
    parser.add_argument("--dynamics_config", help="The path to the dynamics config")
    parser.add_argument("--vhjb_controller_config", help="The path to the config of vhjb controller")
    
    args = parser.parse_args()
    env_name = args.env_name
    dynamics_config = args.dynamics_config
    vhjb_controller_config = args.vhjb_controller_config

    dynamics, nn_policy, model_based_controller = load_systems(env_name, dynamics_config, vhjb_controller_config)

    nn_policy.train()
    
    anim, fig = test_policy(nn_policy, dynamics, model_based_controller)
        
    if isinstance(model_based_controller, LQR):
        P = model_based_controller.P
    elif isinstance(model_based_controller, CartpoleEnergyShapingController):
        _, P = model_based_controller.get_lqr_term()

    visualize_value_landscope_for_lqr(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, 
                                      P, nn_policy.xf, dynamics.states_wrap)

    # xs, costs, dones = next(iter(nn_policy.dataloader))
    # visualize_loss_landscope(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, nn_policy.key,
    #                         xs, partial(nn_policy.hjb_loss), dones=dones)
    # visualize_loss_landscope(nn_policy.value_function_approximator, nn_policy.model_params, nn_policy.model_states, nn_policy.key,
    #                         xs, partial(nn_policy.termination_loss), dones=dones, costs=costs)
    plt.show()
    
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()