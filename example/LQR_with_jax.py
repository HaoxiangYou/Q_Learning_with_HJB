import numpy as np
import scipy.linalg
import jax
import jax.numpy as jnp
import optax
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from flax import linen as nn
from typing import Sequence

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class StatesDataset(Dataset):
    def __init__(self, xs: Sequence[jnp.ndarray]) -> None:
        super().__init__()
        self.xs = xs
    def __len__(self) -> int:
        return len(self.xs)
    def __getitem__(self, index) -> jnp.ndarray:
        return self.xs[index]

class Value(nn.Module):
    hidden_layers: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for hidden_layer in self.hidden_layers:
            x = nn.Dense(hidden_layer)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = jnp.squeeze(jnp.exp(x))
        return x
    
def termination_loss(params, xs, dones, rewards):
    def loss(x, done, reward,):
        v = model.apply(params, x)
        return (v-reward)**2 * done
    return jnp.mean(jax.vmap(loss)(xs, dones, rewards), axis=0)
    
def pVpx(params, x):
    return jax.grad(model.apply, argnums=1)(params, x)

def get_control(pVpx):
    return -jnp.dot(jnp.dot(R_inv,B.T), pVpx) / 2

def HJB_loss(params, xs, dones):
    def loss(x, done):
        v_gradient = pVpx(params, x)
        u = get_control(v_gradient)
        vdot = jnp.dot(v_gradient, jnp.dot(A, x) + jnp.dot(B, u))
        loss = jnp.square(vdot + jnp.dot(jnp.dot(u, R), u) + jnp.dot(jnp.dot(x, Q),x))
        return loss * (1-done)
    return jnp.mean(jax.vmap(loss)(xs, dones), axis=0)

def total_loss(params, xs, dones, rewards):
    return HJB_loss(params, xs, dones) + regulation * termination_loss(params, xs, dones, rewards)

@jax.jit
def update(params, xs, dones, rewards):
    grad = jax.grad(total_loss)(params, xs, dones, rewards)
    return jax.tree_map(lambda p,g: p - learning_rate * g, params, grad)

A = np.array([[0,1],[0,0]], dtype=np.float32)
B = np.array([[0],[1]], dtype=np.float32)
Q = np.array([[1,0],[0,1]], dtype=np.float32)
R = np.array([[1]], dtype=np.float32)
R_inv = np.linalg.inv(R)
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

key = jax.random.PRNGKey(0)
torch.random.manual_seed(0)

# Hyperparameters
regulation = 1e-3
hidden_layers = [100,100]
batch_size = 256
epochs = 1000
learning_rate = 1e-3
xs_sample_size = 10000
xs_success_sample_size = 1000
x_train_range = 2
x_test_range = 1
x_fail_range = 1.9
fail_reward = 10
x_success_range = 1e-3
    
# Initialize the model
key, key_to_use = jax.random.split(key)
x = jax.random.uniform(key_to_use, (2,))
model = Value(hidden_layers=hidden_layers)
key, key_to_use = jax.random.split(key)
params = model.init(key_to_use, x)
optimizer = optax.adam(learning_rate=learning_rate)

# Setup the dataset
states = [np.random.rand(2,) * x_train_range * 2 - x_train_range for _ in range(xs_sample_size)]
states.extend([np.random.rand(2,) * x_success_range * 2 - x_success_range for _ in range(xs_success_sample_size)])
dones = []
termination_rewards = []
for state in states:
    if np.any(np.abs(state) > x_test_range):
        dones.append(1.0)
        termination_rewards.append(fail_reward)
    elif np.linalg.norm(state) < x_success_range:
        dones.append(1.0)
        termination_rewards.append(0)
    else:
        dones.append(0.0)
        termination_rewards.append(0)

xs_train = [(states, done, termination_reward) for states, done, termination_reward in zip(states, dones, termination_rewards)]

train_dataset = StatesDataset(xs_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)

# Training
for epoch in range(epochs):
    for i, (xs, dones, rewards) in enumerate(train_dataloader):
        params = update(params, xs, dones, rewards)
    if epoch % 100 == 0:
        hjb_loss = HJB_loss(params, xs, dones)
        done_loss = termination_loss(params, xs, dones, rewards)
        print(f"epoch: {epoch}, hjb loss:{hjb_loss:.5f}, termination loss:{done_loss:.5f}")

# Testing
timestep = 0.01
time_span = torch.arange(0,10,timestep)
x0 = np.random.rand(2,) * x_test_range * 2 - x_test_range
xs_learned = [x0]
xs_lqr = [x0]
us_learned = []
us_lqr = []
costs_learned = []
costs_lqr = []
cost_learned = 0
cost_lqr = 0
for _ in time_span:
    x_learned = xs_learned[-1]
    x_lqr = xs_lqr[-1]

    u_learned = get_control(pVpx(params, x_learned))
    u_lqr = -R_inv @ B.T @ P @ x_lqr

    cost_learned += (x_learned.T @ Q @ x_learned + u_learned.T @ R @ u_learned).item() * timestep
    cost_lqr += (x_lqr.T @ Q @ x_lqr + u_lqr.T @ R @ u_lqr)* timestep

    us_learned.append(u_learned)
    us_lqr.append(u_lqr)

    costs_learned.append(cost_learned)
    costs_lqr.append(cost_lqr)

    xs_learned.append((A @ x_learned + B @ u_learned)*timestep + x_learned)
    xs_lqr.append((A @ x_lqr + B @ u_lqr)*timestep + x_lqr)

xs_learned = np.array(xs_learned)
xs_lqr = np.array(xs_lqr)
us_learned = np.array(us_learned)
us_lqr = np.array(us_lqr)


x1 = np.arange(-x_train_range, x_train_range, 0.05)
x2 = np.arange(-x_train_range, x_train_range, 0.05)

X1, X2 = np.meshgrid(x1, x2)
v_learned = np.zeros_like(X1)
v_lqr = np.zeros_like(X1)

quick_apply = jax.jit(model.apply)

for i in range(X1.shape[0]):
    for j in range(X2.shape[0]):
        x = np.array([X1[i,j], X2[i,j]])
        v_learned[i,j] = quick_apply(params, x)
        v_lqr[i,j] = x.T @ P @ x

plt.figure(1)
for i in range(xs_learned.shape[1]):
    plt.plot(time_span, xs_learned[:-1,i], label=f"learned x[{i}]")
    plt.plot(time_span, xs_lqr[:-1,i], label=f"lqr x[{i}]")
plt.title("states vs time")
plt.xlabel("time")
plt.ylabel("state")
plt.legend()

plt.figure(2)
for i in range(us_learned.shape[1]):
    plt.plot(time_span, us_learned[:,i], label=f"learned u[{i}]")
    plt.plot(time_span, us_lqr[:,i], label=f"lqr u[{i}]")
plt.title("input vs time")
plt.xlabel("time")
plt.ylabel("input")
plt.legend()

plt.figure(3)
plt.plot(time_span, costs_learned, label="learned")
plt.plot(time_span, costs_lqr, label="lqr")
plt.xlabel("time")
plt.ylabel("cumulated cost")
plt.title("cumulated cost vs time")

plt.legend()

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, v_lqr)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('value')
plt.title("LQR value function levelset")

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, v_learned)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('value')
plt.title("Learned value function levelset")

plt.show()

import pdb; pdb.set_trace()


