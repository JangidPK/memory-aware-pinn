"""
training/trainer.py

Two trainers:

1. BaselineTrainer
   Trains an ordinary PINN without memeory

2. MemoryPINNTrainer
   Trains the PINN + kernel network jointly (or with a fixed
   kernel)
"""

import torch

from physics.diffusion import grad, memory_residual
from physics.losses import ic_loss, bc_loss, data_loss, total_loss


class BaselineTrainer:
    """ordinary PINN, dc/dt = D * c_xx."""

    def __init__(self, pinn, D=0.1, lr=1e-3, device="cpu"):
        self.pinn = pinn.to(device)
        self.D = D
        self.device = device
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        self.history = {"loss": [], "physics": [], "ic": [], "bc": []}


    def _physics_residual(self, x, t):
        c = self.pinn(x, t)
        c_t = grad(c, t)
        c_x = grad(c, x)
        c_xx = grad(c_x, x)
        R = c_t - self.D * c_xx
        return R
    

    def train(self, x_colloc, t_colloc, x_ic, c0_ic, t_bc, epochs=5000,
              print_every=500):
        
        x_colloc = x_colloc.clone().requires_grad_(True)
        t_colloc = t_colloc.clone().requires_grad_(True)

        for epoch in range(epochs):

            self.optimizer.zero_grad()

            R = self._physics_residual(x_colloc, t_colloc)
            l_phys = torch.mean(R ** 2)
            l_ic = ic_loss(self.pinn, x_ic, c0_ic)
            l_bc = bc_loss(self.pinn, t_bc)

            loss = l_phys + 10.0 * l_ic + 10.0 * l_bc
            loss.backward()
            self.optimizer.step()

            self.history["loss"].append(loss.item())
            self.history["physics"].append(l_phys.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())

            if print_every and epoch % print_every == 0:
                print(f"[Baseline] epoch {epoch:5d} | loss {loss.item():.3e} "
                      f"| physics {l_phys.item():.3e} | ic {l_ic.item():.3e} "
                      f"| bc {l_bc.item():.3e}")

        return self.history


class MemoryPINNTrainer:
    """Pmemory PINN + (optional) kernel network."""

    def __init__(self, pinn, kernel_net, D=0.1, lr=1e-3, device="cpu",
                 learn_kernel=True):
        

        self.pinn = pinn.to(device)
        self.kernel_net = kernel_net.to(device)
        self.D = D
        self.device = device
        self.learn_kernel = learn_kernel

        params = list(self.pinn.parameters())
        if learn_kernel:
            params += list(self.kernel_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.history = {"loss": [], "physics": [], "ic": [], "bc": [], "data": []}







    def train(self, x_grid, t_grid, x_ic, c0_ic, t_bc,
              x_obs, t_obs, c_obs, 
              epochs=5000, print_every=500):
        """
        x_grid, t_grid : (Nx,), (Nt,) tensors -- shared collocation grid
        x_ic, c0_ic    : (N,1) tensors -- initial condition points
        t_bc           : (N,1) tensor -- boundary time points
        x_obs, t_obs, c_obs : (N,1) tensors -- sparse observation data
        """
        for epoch in range(epochs):
            
            self.optimizer.zero_grad()

            R, c_grid = memory_residual(self.pinn, self.kernel_net,
                                         x_grid, t_grid, D=self.D)
            l_phys = torch.mean(R ** 2)
            l_ic = ic_loss(self.pinn, x_ic, c0_ic)
            l_bc = bc_loss(self.pinn, t_bc)
            l_data = data_loss(self.pinn, x_obs, t_obs, c_obs)

            loss = total_loss(l_phys, l_ic, l_bc, l_data)
            loss.backward()
            self.optimizer.step()

            self.history["loss"].append(loss.item())
            self.history["physics"].append(l_phys.item())
            self.history["ic"].append(l_ic.item())
            self.history["bc"].append(l_bc.item())
            self.history["data"].append(l_data.item())

            if print_every and epoch % print_every == 0:
                print(f"[Memory] epoch {epoch:5d} | loss {loss.item():.3e} "
                      f"| physics {l_phys.item():.3e} | ic {l_ic.item():.3e} "
                      f"| bc {l_bc.item():.3e} | data {l_data.item():.3e}")

        return self.history
