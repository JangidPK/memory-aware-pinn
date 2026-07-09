
import torch

from physics.quadrature import trapezoidal_memory_integral


def grad(outputs, inputs):
    """Convenience wrapper around torch.autograd.grad 
    for scalar-per-row outputs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]


def compute_derivatives_on_grid(pinn, x_grid, t_grid):
    """
    Evaluate c, dc/dt and d^2c/dx^2 on the full (Nx, Nt) grid.

    Parameters
    ----------
    pinn : PINN model, forward(x, t) -> c
    x_grid : (Nx,) tensor
    t_grid : (Nt,) tensor

    Returns
    -------
    c    : (Nx, Nt) tensor
    c_t  : (Nx, Nt) tensor
    c_xx : (Nx, Nt) tensor
    """
    
    Nx, Nt = len(x_grid), len(t_grid)
    X, T = torch.meshgrid(x_grid, t_grid, indexing="ij")  # (Nx, Nt)

    x = X.reshape(-1, 1).clone().requires_grad_(True)
    t = T.reshape(-1, 1).clone().requires_grad_(True)

    # pinn takes value of x and t, returns c

    c = pinn(x, t)                       # (Nx*Nt, 1)

    # calculate time derivative of concentration
    c_t = grad(c, t)                     # dc/dt

    # caclcuate the position derivative
    c_x = grad(c, x)                     # dc/dx

    # second position derivative
    c_xx = grad(c_x, x)                  # d^2c/dx^2

    c = c.reshape(Nx, Nt)
    c_t = c_t.reshape(Nx, Nt)
    c_xx = c_xx.reshape(Nx, Nt)

    return c, c_t, c_xx


def memory_residual(pinn, kernel_net, x_grid, t_grid, D=0.1):
    """
    Physics residual for the non-Markovian diffusion equation (Phase 7):

        R = dc/dt - int_0^t K(t-s) * D * c_xx(x,s) ds

    Returns
    -------
    R : (Nx, Nt) tensor -- residual at every collocation point
    c : (Nx, Nt) tensor -- predicted concentration (reused for other losses)
    """

    # calculate function c(x,t) its first derivative wrt time and second wrt position
    c, c_t, c_xx = compute_derivatives_on_grid(pinn, x_grid, t_grid)


    dt = t_grid[1] - t_grid[0]
    
    # full laplacian L(x, t)
    laplacian_term = D * c_xx  # (Nx, Nt)

    I = trapezoidal_memory_integral(kernel_net, t_grid, laplacian_term, dt)

    # the discrete convolution, I, in the documentation

    R = c_t - I
    
    # residual error , this is physics loss

    return R, c
