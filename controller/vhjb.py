import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, List, Tuple, Union
from collections import deque
from functools import partial
from controller.controller import Controller
from dynamics.dynamics import Dynamics
from configs.controller.vhjb_controller_config import VHJBControllerConfig
from utils.utils import np_collate

class ValueFunctionApproximator(nn.Module):
    features: Sequence[int]
    mean: jnp.ndarray
    std: jnp.ndarray
    # a flag to indicate whether to use batch norm layer
    # the flag should remain unchange after initialization
    # TODO write as a property
    using_batch_norm: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.float32:
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            axis_name="batch"
        )

        x = (x - self.mean) / self.std

        for i, feature in enumerate(self.features):
            x = nn.Dense(feature)(x)

            # apply nonlinearity if not the last feature
            if not i == len(self.features) - 1:
                if self.using_batch_norm:
                    x = norm()(x)
                x = nn.relu(x)
        # dot product for last axis
        # assume the value function can be approximate by some quadratic form
        x = jnp.einsum('...i,...i->...', x, x)

        return x

class StatesDataset(Dataset):
    def __init__(self, xs:List[Tuple[jnp.ndarray, float, float]], max_size:int) -> None:
        """
        Each elements in xs is a tuple of states, input, cost, done
        """
        super().__init__()
        self.xs = deque(maxlen=max_size)
        self.xs.extend(xs)
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, index):
        return self.xs[index]

class VHJBController(Controller):

    def __init__(self, dynamics:Dynamics, config: VHJBControllerConfig) -> None:
        super().__init__()

        # Seed
        self.key = jax.random.PRNGKey(config.seed)
        torch.random.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Load system
        self.dynamics = dynamics
        self.state_dim, self.control_dim = self.dynamics.get_dimension()
        self.umin, self.umax = self.dynamics.get_control_limit()
        assert self.umin.shape[0] == self.control_dim
        assert self.umax.shape[0] == self.control_dim

        self.Q = config.Q
        self.R = config.R
        self.R_inv = jnp.linalg.inv(self.R)
        self.xf = config.xf
        self.obs_min = config.obs_min
        self.obs_max = config.obs_max

        # Compute additional informations for the system
        self.system_additional_init()

        # Initial model
        self.value_function_approximator = ValueFunctionApproximator(
            features=config.features, 
            mean=config.normalization_mean, 
            std=config.normalization_std,
            using_batch_norm=config.using_batch_norm)
        self.key, key_to_use = jax.random.split(self.key)
        self.train_mode = False
        model_variables = self.value_function_approximator.init(key_to_use, jnp.zeros((1,self.state_dim)), train=self.train_mode)
        self.model_states, self.model_params = model_variables.pop('params')
        del model_variables
        self.optimizer = optax.adam(learning_rate=config.lr)
        self.optimizer_states = self.optimizer.init(self.model_params)
        self.loss_fn = jnp.abs
        regularization_scheduler_configs = [{"init_value": config.regularization_init_value, "end_value": config.regularization_end_value,
                                             "peak_value": config.regularization_peak_value, "warmup_steps":config.regularization_warmup_steps_per_cycle, 
                                             "decay_steps": config.regularization_total_steps_per_cycle,}] * config.regularization_num_of_cycles
        self.regularization_scheduler = jax.jit(optax.sgdr_schedule(regularization_scheduler_configs))
        self.update_counter = jnp.array(0)
        self.regularization = self.regularization_scheduler(self.update_counter)
        self.epochs = config.epochs
        self.batch_size = config.batch_size

        # Initial dataset and reply buffer
        self.maximum_timestep = config.maximum_step
        self.num_of_trajectories_per_epoch = config.num_of_trajectories_per_epoch

        interior_states = [self.dynamics.states_wrap(np.random.uniform(low=-1, high=1, size=self.state_dim) * 
                            config.interior_states_std + config.interior_states_mean)
                           for _ in range(config.num_of_interior_data)]
        interior_dones = [0] * config.num_of_interior_data
        interior_costs = [0] * config.num_of_interior_data

        boundary_states = [self.dynamics.states_wrap(np.random.uniform(low=-1, high=1, size=self.state_dim) * 
                            config.boundary_states_std + config.boundary_states_mean)
                           for _ in range(config.num_of_boundary_data)]
        boundary_dones = [1] * config.num_of_boundary_data
        boundary_costs = [min(self.termination_cost(x), config.boundary_cost_clip) for x in boundary_states]

        data = list(zip(interior_states, interior_costs, interior_dones))
        data.extend(list(zip(boundary_states, boundary_costs, boundary_dones)))

        self.replay_buffer = StatesDataset(data, config.maximum_buffer_size)
        # The drop last was true to prevent jax jit recompile over and over again 
        # because the size of last batch may change due to replay buffer change
        self.dataloader = DataLoader(self.replay_buffer, batch_size=self.batch_size, shuffle=True, collate_fn=np_collate, drop_last=True)

    def system_additional_init(self) -> None:
        # Linearized the dynamics around the xf,
        # Assume f(xf,0) = 0, which means xf is an equilibrium points with 0 input
        uf = np.zeros((self.control_dim,), dtype=np.float32)
        Alin, Blin = jax.jacobian(jax.jit(self.dynamics.dynamics_step), argnums=[0,1])(self.xf, uf)
        self.P = scipy.linalg.solve_continuous_are(Alin, Blin, self.Q, self.R)

    def running_cost(self, x: Union[np.ndarray, jnp.ndarray], u:Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        x_diff = self.dynamics.states_wrap(x - self.xf)
        return x_diff.T @ self.Q @ x_diff + u.T @ self.R @ u

    def termination_cost(self, x: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        x_diff = self.dynamics.states_wrap(x-self.xf)
        return x_diff.T @ self.P @ x_diff
    
    def rollout_trajectory(self) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        trajectory = []
        x = self.dynamics.get_initial_state()
        done = 0.0
        for i in range(self.maximum_timestep):
            if np.any(x > self.obs_max) or np.any(x < self.obs_min):
                done = 1.0
                cost = self.termination_cost(x)
                trajectory.append((x, cost, done)) 
                break
            else:
                u = self.get_control_efforts(x)
                cost = self.running_cost(x, u) * self.dynamics.dt
                trajectory.append((x, cost, done)) 
                x = self.dynamics.simulate(x,u)

        if done == 0.0:
            cost = self.termination_cost(x)
            done = 1.0
            trajectory.append((x, cost, done))

        return trajectory
    
    def get_trajectory_cost(self, trajectory):
        total_cost = 0.0
        for x, cost, done in trajectory:
            total_cost += cost
        return total_cost

    def get_v_gradient(self, params, states, x):
        return jax.grad(self.value_function_approximator.apply, argnums=1, has_aux=True)({'params': params, **states}, x, train=self.train_mode, mutable=list(states.keys()))

    @partial(jax.jit, static_argnums=(0,))
    def get_control_efforts_with_additional_term(self, params, states, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        This function give the optimal input given, value function params, state.
        The jit decorator is used to accelerate the computing

        Args:
            params: model params for neural networks
            states: model states such as batch norm statistics for neural networks
            x: state
        
        Returns:
            a tuple of optimal u and v_gradient and updated model states
        """
        v_gradient, updated_states = self.get_v_gradient(params, states, x)
        f_1, f_2 = self.dynamics.get_control_affine_matrix(x)
        u = -self.R_inv @ f_2.T @ v_gradient / 2
        return u, v_gradient, updated_states
    
    def get_control_efforts(self, x):
        u, v_gradient, updated_states = self.get_control_efforts_with_additional_term(self.model_params, self.model_states, x)
        return np.asarray(u)
    
    def hjb_loss(self, params, states, xs, dones):
        def loss(x, done):
            f_1, f_2 = self.dynamics.get_control_affine_matrix(x)
            u, v_gradient, updated_states = self.get_control_efforts_with_additional_term(params, states, x)
            x_dot = f_1 + f_2 @ u
            v_dot = v_gradient.T @ x_dot
            loss = self.loss_fn(v_dot + self.running_cost(x, u)) * (1-done)
            return loss, updated_states
        batch_losses, updated_states = jax.vmap(
            loss, out_axes=(0, None), axis_name='batch'
        )(xs, dones)

        # the mean is calculated based on the number of "dones" to prevent 
        # losses coupling with the proportion of boundary and interior data   
        return jnp.sum(batch_losses, axis=0) / (jnp.sum(1-dones) + 1e-10), updated_states
    
    def termination_loss(self, params, states, xs, dones, costs):
        def loss(x, done, cost):
            v, updated_states = self.value_function_approximator.apply({'params': params, **states}, x, train=self.train_mode, mutable=list(states.keys()))
            return self.loss_fn(v-cost) * done, updated_states
        batch_losses, updated_states = jax.vmap(
            loss, out_axes=(0, None), axis_name='batch'
        )(xs, dones, costs)

        # the mean is calculated based on the number of "dones" to prevent 
        # losses coupling with the proportion of boundary and interior data   
        return jnp.sum(batch_losses, axis=0) / (jnp.sum(dones) + 1e-10), updated_states

    @partial(jax.jit, static_argnums=(0,))
    def params_update(self, params, states, optimizer_state, xs, dones, costs, regularization):
        """
        Apply one step update to the neural network params and states.
        
        Note, every attribute may change in the function should pass explicitly as arguments, 
        because the jit will treat the self object as immutable dictionary, 
        TODO a better implemmentation will build entire class as pytree
        see:https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        Args:
            params: params for neural network
            states: state params such as batch norm statistics for nerual network
            optimizer_state: state params for optimizer
            xs: a batch of states
            dones: a batch of flags
            costs: a batch termination costs for each states, this is only useful when done flag is true
            regularization: the hyperparameters to balance the termination cost and hjb cost
       
        Returns:
            params: updated neural network params
            updated_states: updated neural network state params
            optimizer_state: updated optimizer state
            total_loss: hjb_loss + regularization * termination_loss
            hjb_loss: loss for hjb (interior loss for pde)
            termination_loss, loss for termination states (boundary conditions)
        """
        (hjb_loss, updated_states), hjb_grad = jax.value_and_grad(self.hjb_loss, has_aux=True)(params, states, xs, dones)
        (termination_loss, updated_states), termination_grad = jax.value_and_grad(self.termination_loss, has_aux=True)(params, states, xs, dones, costs)
        total_loss = hjb_loss + regularization * termination_loss
        total_grad = jax.tree_util.tree_map(lambda g1, g2: g1 + regularization * g2, hjb_grad, termination_grad)
        updates, optimizer_state = self.optimizer.update(total_grad, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, updated_states, optimizer_state, total_loss, hjb_loss, termination_loss
    
    def train(self):

        for epoch in range(self.epochs):
            # sample additional trajectory
            trajectory_costs = 0
            self.train_mode = False
            for _ in range(self.num_of_trajectories_per_epoch):
                trajectory = self.rollout_trajectory()
                trajectory_costs += self.get_trajectory_cost(trajectory)
                self.replay_buffer.xs.extend(trajectory)
            # fit the value function
            total_losses = 0
            hjb_losses = 0
            termination_losses = 0
            self.train_mode = True
            for i, (xs, costs, dones) in enumerate(self.dataloader):
                self.model_params, self.model_states, self.optimizer_states, total_loss, hjb_loss, termination_loss = self.params_update(
                    self.model_params, self.model_states, self.optimizer_states, xs, dones, costs, self.regularization
                )
                total_losses += total_loss
                hjb_losses += hjb_loss
                termination_losses += termination_loss

                # update the regularization
                self.update_counter = jax.jit(optax._src.numerics.safe_int32_increment)(self.update_counter)
                self.regularization = self.regularization_scheduler(self.update_counter)

            if (epoch+1) % 10 == 0:
                if self.num_of_trajectories_per_epoch > 0:
                    print(f"epoch:{epoch+1}, average trajectory cost:{trajectory_costs/self.num_of_trajectories_per_epoch:2f}")
                print(f"epoch:{epoch+1}, total loss:{total_losses/len(self.dataloader):.5f}, regulation: {self.regularization:.1f},"\
                      f"hjb loss:{hjb_losses/len(self.dataloader):.5f}, termination loss:{termination_losses/len(self.dataloader):.5f}")