import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from common.dynamics.dynamics import Dynamics
from common.controller.controller import Controller
from typing import List

class VGradient(nn.Module):    
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x:torch.Tensor):
        pVpx = self.net(x)
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
    def __init__(self, xs:List[np.ndarray]) -> None:
        super().__init__()
        self.xs = xs
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, index):
        return self.xs[index].astype(np.float32)

class VGradientPolicy(Controller):
    
    def __init__(self, dynamics:Dynamics, Q:np.ndarray, R:np.ndarray) -> None:
        super().__init__()

        # Some hyperparameters
        seed = 0
        hidden_size = 100
        lr=1e-5
        self.batch_size = 64
        weight_decay=0
        momentum=0.9
        self.warm_up_epochs = 10
        self.epochs = 50
        self.xs_range = np.array([np.pi, np.pi, 5, 5])
        self.sample_size = 10000

        self.Q = torch.tensor(Q).to(torch.float32)
        self.R = torch.tensor(R).to(torch.float32)

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load system
        self.dynamics = dynamics
        self.state_dim, self.control_dim = self.dynamics.get_dimension()
        self.VGradient = VGradient(self.state_dim, hidden_size)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.VGradient.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        # Initial dataset
        self.expert_dataset = ExpertDataset([],[])
        xs = [np.random.rand(self.state_dim,) * self.xs_range *2 - self.xs_range for _ in range(self.sample_size)]
        self.states_dataset = StatesDataset(xs)

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
        
    def get_control_efforts(self, xs:torch.Tensor):
        """
        The control input is given by:
            u = \ underset{u}{argmin } \ nabla V(x)^T f(x,u) + x^T Q x + u^T R u
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
        return us.squeeze(2)
    
    def HJB_loss(self, xs, us):
        """
            Calculate HJB loss given the xs and us.\,
            The HJB loss is defined by the L2 norm of:
                pVpx^T @ f(x,u) + u^T @ R @ u +x^T @ Q @ x
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
        xs_cost = torch.einsum('bi,ij,bj->b', xs, self.Q, xs)

        loss = self.loss_fn(Vdots + us_cost, -xs_cost)
        return loss

    def train(self):

        expert_dataloader = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
        states_dataloader = DataLoader(self.states_dataset, batch_size=self.batch_size, shuffle=True)

        self.VGradient.train()

        for epoch in range(self.epochs):
            running_loss = 0
            dataset_size = 0
            
            # Use expert data to warm-up the VG
            if epoch <= self.warm_up_epochs:
                dataset_size = len(expert_dataloader)
                for xs ,us in expert_dataloader:
                    loss = self.HJB_loss(xs,us)
                    running_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Only use HJB loss function continue to train
            else:
                dataset_size = len(states_dataloader)
                for xs in states_dataloader:
                    us = self.get_control_efforts(xs)
                    loss = self.HJB_loss(xs,us)
                    running_loss += loss.item()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    dataset_size += 1 
            
            print(f"epoch {epoch+1}, loss {running_loss / dataset_size}")