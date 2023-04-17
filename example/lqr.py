import numpy as np
import matplotlib.pyplot as plt
import torch
import gin
import os
from common.dynamics.linear import LinearDynamics
from common.controller.lqr import LQR
from common.controller.pVpx_policy import VGradientPolicy
from common.configs.dynamics.linear_config import LinearDynamicsConfig
from common.configs.controller.pVpx_controller_config import pVpxControllerConfig

def test_nn_policy(nn_policy:VGradientPolicy, dynamics:LinearDynamics, x0:np.ndarray, lqr_controller:LQR):
    nn_policy.VGradient.eval()
    t0 = 0
    tf = 10

    t = np.arange(t0, tf, dynamics.dt)
    xs_learned = [x0]
    xs_lqr = [x0]
    us_learned = []
    us_lqr = []
    costs_learned = []
    costs_lqr = []
    cost_learned = 0
    cost_lqr = 0

    with torch.no_grad():
        for _ in t:
            x_learned = xs_learned[-1]
            x_lqr = xs_lqr[-1]
            u_learned = (nn_policy.get_control_efforts(torch.tensor(x_learned.astype(np.float32)).unsqueeze(0))).numpy().squeeze()
            u_lqr = lqr_controller.get_control_efforts(x_lqr)

            us_learned.append(u_learned)
            us_lqr.append(u_lqr)

            cost_learned += nn_policy.get_total_return(x_learned[None,:], u_learned[None,:])
            cost_lqr += nn_policy.get_total_return(x_lqr[None,:], u_lqr[None,:])

            costs_learned.append(cost_learned)
            costs_lqr.append(cost_lqr)

            xs_learned.append(dynamics.simulate(x_learned, u_learned))
            xs_lqr.append(dynamics.simulate(x_lqr, u_lqr))

    xs_learned = np.array(xs_learned)
    xs_lqr = np.array(xs_lqr)
    us_learned = np.array(us_learned)
    us_lqr = np.array(us_lqr)

    plt.figure(1)
    plt.plot(t, xs_learned[:-1,0], label="learned x[0]")
    plt.plot(t, xs_learned[:-1,1], label="learned x[1]")
    plt.plot(t, xs_lqr[:-1,0], label="lqr x[0]")
    plt.plot(t, xs_lqr[:-1,1], label="lqr x[1]")
    plt.title("states vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure(2)
    plt.plot(t, us_learned[:,0], label="learned u[0]")
    plt.plot(t, us_learned[:,1], label="learned u[1]")
    plt.plot(t, us_lqr[:,0], label="lqr u[0]")
    plt.plot(t, us_lqr[:,1], label="lqr u[1]")
    plt.title("input vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure(3)
    plt.plot(t, costs_learned, label="learned")
    plt.plot(t, costs_lqr, label="lqr")
    plt.xlabel("time")
    plt.ylabel("cumulated cost")
    plt.title("cumulated cost vs time")
    plt.legend()

    plt.show()


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
            "../common/configs/controller/linear_pVpx_controller.gin",
        )
    )
    
    gin.parse_config_file(path_to_dynamics_config_file)
    dynamics_config = LinearDynamicsConfig()

    gin.parse_config_file(path_to_controller_config_file)
    controller_config = pVpxControllerConfig()

    return dynamics_config, controller_config

def main():
    dynamics_config, controller_config = load_config()

    np.random.seed(controller_config.seed)
    
    dynamics = LinearDynamics(dynamics_config)
    nn_policy = VGradientPolicy(dynamics, controller_config)
    lqr_controller = LQR(dynamics, controller_config.Q, controller_config.R)

    state_dim, control_dim = dynamics.get_dimension()
    t_span = np.arange(0,5,dynamics_config.dt)

    for i in range(controller_config.num_of_warmup_trajectory):
        
        x0 = controller_config.x0_mean + np.random.rand(state_dim) * 2 * controller_config.x0_std - controller_config.x0_std
        xs_energy_shaping = [x0]
        us_energy_shaping = []
        for _ in range(t_span.shape[0]-1):
            us_energy_shaping.append(lqr_controller.get_control_efforts(xs_energy_shaping[-1]))
            xs_energy_shaping.append(dynamics.simulate(xs_energy_shaping[-1], us_energy_shaping[-1]))

        # Append expert data to the nerual network
        nn_policy.set_expert_set(xs_energy_shaping[:-1], us_energy_shaping)
    
    nn_policy.train()

    x0 = controller_config.x0_mean + np.random.rand(state_dim) * 2 * controller_config.x0_std - controller_config.x0_std

    test_nn_policy(nn_policy, dynamics, x0, lqr_controller)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()