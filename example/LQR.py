# Author: Haoxiang You
"""
We would solve a LQR problem using with:
    min \int_0^inf x^TQx + u^T R u
    s.t. \dot{x} = Ax + Bu 
"""
import torch
import numpy as np
from torch import nn
import scipy.linalg
import matplotlib.pyplot as plt

class VGradient(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # Set the seed
    torch.manual_seed(0)

    # Set up the system
    A = torch.tensor([[1,0],[0,1]],dtype=torch.float32)
    B = torch.tensor([[1,0],[0,1]],dtype=torch.float32)
    Q = torch.tensor([[1,0],[0,1]],dtype=torch.float32)
    R = torch.tensor([[1,0],[0,1]],dtype=torch.float32)
    R_inv = torch.linalg.inv(R)

    input_size = Q.shape[0]
    control_size = R.shape[0]

    # Set hyperparameters:
    train_size = 10000
    hidden_size = 10
    epochs = 10

    # Value function from LQR
    P = torch.tensor(scipy.linalg.solve_continuous_are(A, B, Q, R), dtype=torch.float32)

    # Setup the dataset
    x_train_range = 2
    x_test_range = 1
    train_xs = torch.rand((train_size, input_size)) * x_train_range * 2 - x_train_range

    # Setup network
    model = VGradient(input_size, hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1e-2, momentum=0.9)
    loss_fn = nn.MSELoss()

    noise = 1

    # Train
    for epoch in range(epochs):
        running_loss = 0
        for x in train_xs:
            # Our objective is: \underset{u}{min } \nabla V(x)^T f(x,u) + g(x,u) = 0
            # By first order condition:
            #     u = -R^{-1} B^T/2 \nabla V(x) 
            # So
            #    \nabla V(x)^T(Ax + Bu) + u^T R u = -x^TQx
            pVpx = model(x)


            if epoch == 0:
                # use expert data at the beginning to warm-up
                u = (- R_inv @ B.T @ P @ x) + torch.randn((control_size,)) * noise * 2 - noise
            else:
                u = (- R_inv @ B.T/2 @ pVpx)


            loss = loss_fn(pVpx.T @ (A @ x + B @ u) + u.T @ R @ u, -x.T @ Q @ x)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"epoch {epoch+1}, loss {running_loss / train_size}")

    # Simulate the policy learned:
    timestep = 0.01
    time_span = torch.arange(0,1,timestep)
    x0 = torch.torch.rand(input_size) * x_test_range * 2 - x_test_range
    xs_learned = [x0]
    xs_lqr = [x0]
    us_learned = []
    us_lqr = []
    costs_learned = []
    costs_lqr = []
    cost_learned = 0
    cost_lqr = 0
    with torch.no_grad():
        for _ in time_span:
            x_learned = xs_learned[-1]
            x_lqr = xs_lqr[-1]
            u_learned = -R_inv @ B.T/2 @ model(x_learned)
            u_lqr = -R_inv @ B.T @ P @ x_lqr

            us_learned.append(u_learned)
            us_lqr.append(u_lqr)

            cost_learned += (x_learned.T @ Q @ x_learned + u_learned.T @ R @ u_learned).item() * timestep
            cost_lqr += (x_lqr.T @ Q @ x_lqr + u_lqr.T @ R @ u_lqr).item() * timestep

            costs_learned.append(cost_learned)
            costs_lqr.append(cost_lqr)

            xs_learned.append((A @ x_learned + B @ u_learned)*timestep + x_learned)
            xs_lqr.append((A @ x_lqr + B @ u_lqr)*timestep + x_lqr) 

    # Plots
    xs_learned = np.array([x.numpy() for x in xs_learned])
    xs_lqr = np.array([x.numpy() for x in xs_lqr])
    us_learned = np.array([x.numpy() for x in us_learned])
    us_lqr = np.array([x.numpy() for x in us_lqr])

    plt.figure(1)
    plt.plot(time_span, xs_learned[:-1,0], label="learned x[0]")
    plt.plot(time_span, xs_learned[:-1,1], label="learned x[1]")
    plt.plot(time_span, xs_lqr[:-1,0], label="lqr x[0]")
    plt.plot(time_span, xs_lqr[:-1,1], label="lqr x[1]")
    plt.title("states vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure(2)
    plt.plot(time_span, us_learned[:,0], label="learned u[0]")
    plt.plot(time_span, us_learned[:,1], label="learned u[1]")
    plt.plot(time_span, us_lqr[:,0], label="lqr u[0]")
    plt.plot(time_span, us_lqr[:,1], label="lqr u[1]")
    plt.title("input vs time")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()

    plt.figure(3)
    plt.plot(time_span, costs_learned, label="learned")
    plt.plot(time_span, costs_lqr, label="lqr")
    plt.xlabel("time")
    plt.ylabel("cumulated cost")
    plt.title("cumulated cost vs time")
    plt.legend()

    plt.show()

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()