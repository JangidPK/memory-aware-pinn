
import torch

def trapezoidal_memory_integral(kernel_net, #kernel function
                                 t_grid,    # time grid (kernel depends on time)
                                 L, # full laplacian grid (Nx, Nt)
                                dt  # time step
                                ):

    """
    Parameters
    ----------
    kernel_net : nn.Module, forward(dt_tensor of shape (N,1)) -> (N,1)
    t_grid : (Nt,) tensor of time points (shared across all x)
    L : (Nx, Nt) tensor -- Laplacian term D * c_xx at every grid point
    dt : scalar tensor/float -- uniform time step

    Returns
    -------
    I : (Nx, Nt) tensor -- the memory integral at every grid point
    """
    
    Nx, Nt = L.shape
    device = L.device
    I = torch.zeros((Nx, Nt), device=device, dtype=L.dtype)

    for j in range(Nt):
        if j == 0:
            # integral from 0 to 0 is exactly zero
            continue

        s = t_grid[: j + 1]                 # s_0, ..., s_j
        delta_t = (t_grid[j] - s).reshape(-1, 1)  # (j+1, 1), always >= 0

        # here if the kernel is fixed then kernel weight will be exact
        # however it can also be learned
        K_vals = kernel_net(delta_t).reshape(-1)  # (j+1,)

        weights = torch.ones(j + 1, device=device, dtype=L.dtype)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = weights * dt

        # (Nx, j+1) @ (j+1,) -> (Nx,)
        I[:, j] = L[:, : j + 1] @ (K_vals * weights)

        # This is discrete convolution. 
        # (I_j(x) in the documentation)

    return I
