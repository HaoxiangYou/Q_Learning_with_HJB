import numpy as np
import matplotlib.pyplot as plt
import torch
from common.dynamics.acrobot import Acrobot, p, dt
from common.controller.acrobot_energy_shaping import AcrobotEnergyShapingController
from common.controller.pVpx_policy import VGradientPolicy

def test_nn_policy(nn_policy:VGradientPolicy, acrobot:Acrobot, x0:np.ndarray, energy_shaping_controller=None):
    t0 = 0
    tf = 25

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
            xs[i] = acrobot.simulate(xs[i-1],us[i-1],dt)
            # wrap angle to -np.pi, np.pi
            xs[i,0] = np.arctan2(np.sin(xs[i,0]), np.cos(xs[i,0]))
            xs[i,1] = np.arctan2(np.sin(xs[i,1]), np.cos(xs[i,1]))

    knee = p['l1']*np.cos(xs[:,0]-np.pi/2), p['l1']*np.sin(xs[:,0]-np.pi/2)
    toe = p['l1']*np.cos(xs[:,0]-np.pi/2) + p['l2']*np.cos(xs[:,0]+xs[:,1]-np.pi/2), \
        p['l1']*np.sin(xs[:,0]-np.pi/2) + p['l2']*np.sin(xs[:,0]+xs[:,1]-np.pi/2)

    anim, fig = acrobot.plot_trajectory(t, xs)

    plt.figure(2); 
    plt.clf()
    plt.plot(knee[0], knee[1], 'k.-', lw=0.5, label='knee')
    plt.plot(toe[0], toe[1], 'b.-', lw=0.5, label='toe')
    plt.xlabel('x'); plt.ylabel('y')
    plt.plot(np.linspace(-2,2,100),
             (p['l1']+p['l2'])*np.ones(100), 'r', lw=1,
             label='max height')
    plt.title("position")
    plt.legend()

    plt.figure(3)
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

    np.random.seed(0)

    Q = np.eye(4)
    R = np.eye(1)
    xf = np.array([0, np.pi, 0, 0])

    acrobot = Acrobot()
    nn_policy = VGradientPolicy(acrobot, Q, R, xf)
    energy_shaping_controller = AcrobotEnergyShapingController(acrobot, Q=Q, R=R)

    # Create warmup dataset from energy shaping
    t_span = np.arange(0,5,dt)
    xs_energy_shaping = [np.array([0.001, 0, 0, 0])]
    # xs_energy_shaping = [np.array([np.pi-0.1,0,0.2,0])]
    us_energy_shaping = []
    for i in range(t_span.shape[0]-1):
        us_energy_shaping.append(energy_shaping_controller.get_control_efforts(xs_energy_shaping[-1]))
        xs_energy_shaping.append(acrobot.simulate(xs_energy_shaping[-1], us_energy_shaping[-1], dt))

    # Train the nerual network
    nn_policy.set_expert_set(xs_energy_shaping[:-1], us_energy_shaping)
    nn_policy.train()

    nn_policy.VGradient.eval()

    x0 = np.array([np.pi-0.1,0,0.2,0])
    test_nn_policy(nn_policy=nn_policy, acrobot=acrobot, x0=x0, energy_shaping_controller=energy_shaping_controller)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()