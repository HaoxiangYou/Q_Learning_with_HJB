import torch
import numpy as np
from torch import nn
from collections import deque
from torch.utils.data import Dataset, DataLoader
from common.dynamics.dynamics import Dynamics
from common.controller.controller import Controller
from common.configs.controller.pVpx_controller_config import pVpxControllerConfig
from typing import List

class VGradient(nn.Module):    
    def __init__(self, input_size:int, hidden_size:int, input_mean:np.ndarray, input_range:np.ndarray) -> None:
        super().__init__()

        assert input_mean.ndim == 1
        assert input_range.ndim == 1
        assert input_mean.shape[0] == input_size
        assert input_range.shape[0] == input_size

        self.input_mean = torch.tensor(input_mean).to(torch.float32)
        self.input_range = torch.tensor(input_range).to(torch.float32)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x:torch.Tensor):
        pVpx = self.net((x - self.input_mean) / self.input_range)
        return pVpx
    
class ExpertDataset(Dataset):
    def __init__(self, xs:List[np.ndarray], us:List[np.ndarray]) -> None:
        super().__init__()
        assert len(xs) == len(us)
        self.xs = xs
        self.us = us
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, index):
        return self.xs[index].astype(np.float32), self.us[index].astype(np.float32)
    
class StatesDataset(Dataset):
    def __init__(self, xs:List[np.ndarray], max_states) -> None:
        super().__init__()
        self.xs = deque(maxlen=max_states)
        self.xs.extend(xs)
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, index):
        return self.xs[index].astype(np.float32)

class VGradientPolicy(Controller):
    
    def __init__(self, dynamics:Dynamics, config: pVpxControllerConfig) -> None:
        super().__init__()
        
        # Setup seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Load system
        self.dynamics = dynamics
        self.state_dim, self.control_dim = self.dynamics.get_dimension()
        umin, umax = self.dynamics.get_control_limit()
        assert umin.shape[0] == self.control_dim
        assert umax.shape[0] == self.control_dim
        self.umin = torch.tensor(umin).to(torch.float32)
        self.umax = torch.tensor(umax).to(torch.float32)
        self.Q = torch.tensor(config.Q).to(torch.float32)
        self.R = torch.tensor(config.R).to(torch.float32)
        self.xf = torch.tensor(config.xf).to(torch.float32)

        # Setup nerual network
        self.VGradient = VGradient(self.state_dim, config.hidden_size, config.normalization_mean, config.normalization_std)
        self.loss_fn = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.VGradient.parameters(), lr=config.lr)

        # rollout hyperparameters:
        self.tf = config.max_trajectory_period
        self.rollout_num = config.trajectories_per_rollout
        self.x0_mean = config.x0_mean
        self.x0_std = config.x0_std

        # Training hyperparameters
        self.batch_size = config.batch_size
        self.warm_up_epochs = config.warm_up_epochs
        self.epochs = config.epochs

        # Initial dataset
        self.expert_dataset = ExpertDataset([],[])
        self.states_dataset = StatesDataset([], config.max_states)

    def set_expert_set(self, xs:List[np.ndarray], us:List[np.ndarray]):
        """
        Loading the expert dataset
            
        xs: List of n (self.state,) array
        us: List of n (self.control_dim,) array
        """
        assert len(xs) == len(us)
        assert xs[0].shape[0] == self.state_dim
        assert us[0].shape[0] == self.control_dim

        self.expert_dataset.xs.extend(xs)
        self.expert_dataset.us.extend(us)

        self.states_dataset.xs.extend(xs)
        
    def get_control_efforts(self, xs:torch.Tensor):
        """
        The control input is given by:
            u = \ underset{u}{argmin } \ nabla V(x)^T f(x,u) + delta x^T Q delta x + u^T R u
        Given our system is control affine:
            f(x,u) = f_1(x) + f_2(x) u 
        We can use the first order condition to write control into:
            u = -R^{-1} f_2(x)*T pVpx(x)

        params:
            xs: n X self.state_dim tensor
        returns:
            us: n X self.control_dim tensor
        """
        assert xs.ndim == 2

        f_1, f_2 = self.dynamics.get_control_affine_matrix(xs)
        # f_2 is in n X self.control X self.state_dim, the self.VGradient(xs) is n X self.state_dim. 
        # So I have to increase the dim of pVpx to be n X self.state_dim X 1
        us = -torch.linalg.inv(self.R)/2 @ ( (torch.from_numpy(np.swapaxes(f_2,1,2).astype(np.float32)) @ self.VGradient(xs).unsqueeze(2)))
        us = us.squeeze(2)

        us = torch.clip(us, min=self.umin, max=self.umax)

        return us
    
    def HJB_loss(self, xs, us):
        """
            Calculate HJB loss given the xs and us,
            The HJB loss is defined by :
                pVpx^T @ f(x,u) + u^T @ R @ u + (x-xf)^T @ Q @ (x-xf)
        params:
            xs: n X self.state_dim tensor
            us: n X self.control_dim tensor
        """

        assert xs.ndim == 2
        assert us.ndim == 2
        assert xs.shape[1] == self.state_dim
        assert us.shape[1] == self.control_dim

        pVpx = self.VGradient(xs)
        f_1, f_2 = self.dynamics.get_control_affine_matrix(xs)
        f_1 = torch.tensor(f_1).to(torch.float32)
        f_2 = torch.tensor(f_2).to(torch.float32)

        xdots = f_1 + (f_2 @ us.unsqueeze(2)).squeeze(2) 
        Vdots = torch.bmm(pVpx.unsqueeze(1), xdots.unsqueeze(2)).squeeze()
        us_cost = torch.einsum('bi,ij,bj->b', us, self.R, us)
        
        # Calculate the difference between current state and desired states, and wrap the value to valid range
        xs_diff = self.dynamics.states_wrap(xs-self.xf)

        xs_cost = torch.einsum('bi,ij,bj->b', xs_diff, self.Q, xs_diff)

        loss = self.loss_fn(Vdots + us_cost, -xs_cost)
        return loss
    
    def get_total_return(self, xs:np.ndarray, us:np.ndarray) -> float:
        """
        A coarse estimate of total return, the return is equal to negative cost
        The integral is approximate by addition of:
            (x_diff.T Q x_diff + u.T @ R u) * dt
        """
        xs_diff = self.dynamics.states_wrap(xs-self.xf.numpy())
        xs_cost = np.einsum('bi,ij,bj->b', xs_diff, self.Q.numpy(), xs_diff)
        us_cost = np.einsum('bi,ij,bj->b', us, self.R.numpy(), us)
        return -(np.sum(xs_cost) + np.sum(us_cost)) * self.dynamics.dt

    def rollout_trajectory(self):

        t0 = 0
        t = np.arange(t0, self.tf, self.dynamics.dt)
        total_returns = 0

        self.VGradient.eval()

        for _ in range(self.rollout_num):
            x0 = self.x0_mean + np.random.rand(self.state_dim) * 2 * self.x0_std - self.x0_std
            xs = np.zeros((t.shape[0], self.state_dim))
            us = np.zeros((t.shape[0]-1, self.control_dim))
            xs[0] = x0
            with torch.no_grad():
                for i in range(1, xs.shape[0]):
                    us[i-1] = (self.get_control_efforts(torch.tensor(xs[i-1].astype(np.float32)).unsqueeze(0))).numpy()
                    xs[i] = self.dynamics.simulate(xs[i-1],us[i-1],self.dynamics.dt)
            
            self.states_dataset.xs.extend(xs)
            
            total_returns += self.get_total_return(np.array(xs), np.array(us))

        print(f"Average returns {total_returns/self.rollout_num}")

    def train(self):

        for epoch in range(self.warm_up_epochs):
            expert_dataloader = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
            running_loss = 0
            dataset_size = len(expert_dataloader)
            self.VGradient.train()
            for xs ,us in expert_dataloader:
                loss = self.HJB_loss(xs,us)
                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"warmup epoch {epoch+1}, loss {running_loss / dataset_size}")

        for epoch in range(self.epochs):

            self.rollout_trajectory()

            states_dataloader = DataLoader(self.states_dataset, batch_size=self.batch_size, shuffle=True)
            running_loss = 0
            dataset_size = len(states_dataloader)
            self.VGradient.train()

            for xs in states_dataloader:
                us = self.get_control_efforts(xs)
                loss = self.HJB_loss(xs,us)
                running_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"epoch {epoch+1}, datasize {dataset_size * self.batch_size}, loss {running_loss / dataset_size}")
        