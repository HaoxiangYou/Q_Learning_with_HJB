import numpy as np
import torch
import gin
import os
import matplotlib.pyplot as plt
from common.dynamics.cartpole import Cartpole
from common.controller.cartpole_energy_shaping import CartpoleEnergyShapingController, dt
from common.controller.pVpx_policy import VGradientPolicy
from common.configs.controller.pVpx_controller_config import pVpxControllerConfig

def test_nn_policy(nn_policy:VGradientPolicy, cartpole:Cartpole, x0:np.ndarray, energy_shaping_controller=None):
    t0 = 0
    tf = 10

    t = np.arange(t0, tf, dt)

    xs = np.zeros((t.shape[0], x0.shape[0]))
    us = np.zeros((t.shape[0]-1,))
    if energy_shaping_controller:
        us_energy_shaping = np.zeros_like(us)

    xs[0] = x0
    with torch.no_grad():
        for i in range(1, xs.shape[0]):
            us[i-1] = (nn_policy.get_control_efforts(torch.tensor(xs[i-1].astype(np.float32)).unsqueeze(0))).numpy()
            if energy_shaping_controller:
                us_energy_shaping[i-1] = energy_shaping_controller.get_control_efforts(xs[i-1])
            xs[i] = cartpole.simulate(xs[i-1],us[i-1],dt)
            xs[i] = cartpole.states_wrap(xs[i][None,:])            

    anim, fig = cartpole.plot_trajectory(t,xs)

    plt.figure(2)
    plt.clf()
    plt.plot(t[:-1], us, label="nn_policy")
    if energy_shaping_controller:
        plt.plot(t[:-1], us_energy_shaping, label="energy_shaping")
    plt.xlabel('t') 
    plt.ylabel('u')
    plt.legend()
    plt.title("input")

    plt.show()
    plt.close()

    return xs

def main():

    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../common/configs/controller/cartpole_pVpx_controller.gin",
        )
    )
    gin.parse_config_file(path_to_config_file)
    controller_config = pVpxControllerConfig()

    np.random.seed(controller_config.seed)

    cartpole = Cartpole()
    nn_policy = VGradientPolicy(cartpole, controller_config)
    energy_shaping_controller = CartpoleEnergyShapingController(cartpole, Q=controller_config.Q, R=controller_config.R)

    # Create warmup dataset from energy shaping
    t_span = np.arange(0,5,dt)

    for i in range(controller_config.num_of_warmup_trajectory):
        
        x0 = controller_config.x0_mean + np.random.rand(4) * 2 * controller_config.x0_std - controller_config.x0_std
        xs_energy_shaping = [x0]
        us_energy_shaping = []
        for _ in range(t_span.shape[0]-1):
            us_energy_shaping.append(energy_shaping_controller.get_control_efforts(xs_energy_shaping[-1]))
            xs_energy_shaping.append(cartpole.simulate(xs_energy_shaping[-1], us_energy_shaping[-1], dt))

        # Append expert data to the nerual network
        nn_policy.set_expert_set(xs_energy_shaping[:-1], us_energy_shaping)
    
    nn_policy.train()
    nn_policy.VGradient.eval()

    x0 = controller_config.x0_mean + np.random.rand(4) * 2 * controller_config.x0_std - controller_config.x0_std

    test_nn_policy(nn_policy, cartpole, x0, energy_shaping_controller)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()