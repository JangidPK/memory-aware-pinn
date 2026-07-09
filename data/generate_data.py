
import numpy as np

def gaussian_pulse(x, x0=0.5, sigma=0.05):
    """Initial condition: a Gaussian centered at x0."""
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def generate_ordinary_diffusion(D=0.1, 
                            Nx=101, Nt=201, L=1.0, T=1.0,
                            x0=0.5, sigma=0.05):
    """
    Solving difffusin equation using finite difference method
    with zero Dirichlet boundary conditions.

    Returns
    -------
    x : (Nx,) array
    t : (Nt,) array
    c : (Nx, Nt) array -- c[i, j] = concentration at x_i, t_j
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = D * dt / dx ** 2 
    # for this simulation remains numerically stable
    # Von Neumann stability
    if r > 0.5:
        raise ValueError(
            f"Stability violated: r = D*dt/dx^2 = {r:.3f} > 0.5. "
            f"Reduce dt or increase dx."
        )


    c = np.zeros((Nx, Nt))
    c[:, 0] = gaussian_pulse(x, x0, sigma)
    c[0, 0] = 0.0
    c[-1, 0] = 0.0

    for n in range(0, Nt - 1):
        lap = np.zeros(Nx)

        # calculate laplacian 
        lap[1:-1] = (c[2:, n] - 2 * c[1:-1, n] + c[:-2, n]) / dx ** 2

        # integrate wrt time c(t) = c(t-dt) + Laplacian * D * dt 
        c[:, n + 1] = c[:, n] + dt * D * lap
        c[0, n + 1] = 0.0
        c[-1, n + 1] = 0.0

    return x, t, c


def generate_memory_diffusion(kernel_fn, D=0.1, Nx=101, Nt=201, L=1.0, T=1.0,
                               x0=0.5, sigma=0.05):
    """
    Solver for the memory diffusion equation

    Parameters
    ----------
    kernel_fn : callable
    Returns
    -------
    x, t, c  (same shapes as generate_ordinary_diffusion)
    """


    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]


    c = np.zeros((Nx, Nt))
    c[:, 0] = gaussian_pulse(x, x0, sigma)
    c[0, 0] = 0.0
    c[-1, 0] = 0.0



    # history of D * laplacian(c) at every past time step
    lap_history = np.zeros((Nx, Nt))

    def laplacian(col):
        lap = np.zeros(Nx)
        lap[1:-1] = (col[2:] - 2 * col[1:-1] + col[:-2]) / dx ** 2
        return lap

    lap_history[:, 0] = D * laplacian(c[:, 0])

    for n in range(0, Nt - 1):


        t_n = t[n]
        s = t[: n + 1]
        weights = np.ones_like(s)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights *= dt

        # kernel
        K_vals = kernel_fn(t_n - s) 

        integral = lap_history[:, : n + 1] @ (K_vals * weights)  # (Nx,)

        #integrate in middle
        c[:, n + 1] = c[:, n] + dt * integral

        #impose bounds
        c[0, n + 1] = 0.0
        c[-1, n + 1] = 0.0

        # add one more history for all positions
        lap_history[:, n + 1] = D * laplacian(c[:, n + 1])

    return x, t, c


def subsample_observations(x, t, c, fraction=0.1, noise_std=0.0, seed=0):
    """
    Randomly subsample the (x, t) grid to produce data,
    optionally with additive Gaussian noise
    
    Returns
    -------
    x_obs, t_obs, c_obs : 1D arrays of the same length
    """

    rng = np.random.default_rng(seed)

    X, T = np.meshgrid(x, t, indexing="ij")
    X = X.flatten()
    T = T.flatten()
    C = c.flatten()

    n_total = len(C)
    n_obs = max(1, int(fraction * n_total))
    idx = rng.choice(n_total, size=n_obs, replace=False)

    x_obs, t_obs, c_obs = X[idx], T[idx], C[idx]

    if noise_std > 0:
        
        c_obs = c_obs + rng.normal(0, noise_std, size=c_obs.shape)

    return x_obs, t_obs, c_obs
