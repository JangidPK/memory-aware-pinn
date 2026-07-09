
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import (
    generate_memory_diffusion, gaussian_pulse, subsample_observations
)
from models.pinn import PINN
from models.kernel_network import KernelNet, FixedExponentialKernel
from training.trainer import MemoryPINNTrainer
from utils.metrics import mse, relative_l2_error
from utils.plotting import plot_heatmaps, plot_kernel, plot_training_curves


TRUE_KERNEL_FN = lambda dt: np.exp(-dt)          
# ground-truth kernel used to generate data

TRUE_KERNEL_TORCH = lambda dt: torch.exp(-dt)    

def run_experiment(name, learn_kernel=True, obs_fraction=0.1, noise_std=0.0,
                    D=0.01, Nx=41, Nt=41, L=1.0, T=1.0, epochs=5000,
                    lr=1e-3, device="cpu", out_dir="results", print_every=500, 
                    r_seed = 304):


    torch.manual_seed(r_seed)
    np.random.seed(r_seed)
    
    
    os.makedirs(out_dir, exist_ok=True)


    # synthetic ground truth 
    x_np, t_np, c_np = generate_memory_diffusion(
        kernel_fn=TRUE_KERNEL_FN, D=D, Nx=Nx, Nt=Nt, L=L, T=T
    )


    x_grid = torch.tensor(x_np, dtype=torch.float32, device=device)
    t_grid = torch.tensor(t_np, dtype=torch.float32, device=device)

    x_ic = x_grid.reshape(-1, 1)
    c0_ic = torch.tensor(gaussian_pulse(x_np), dtype=torch.float32,
                          device=device).reshape(-1, 1)
    c0_ic[0] = 0.0
    c0_ic[-1] = 0.0

    t_bc = t_grid.reshape(-1, 1)

    x_obs_np, t_obs_np, c_obs_np = subsample_observations(
        x_np, t_np, c_np, fraction=obs_fraction, noise_std=noise_std, seed=0
    )
    x_obs = torch.tensor(x_obs_np, dtype=torch.float32, device=device).reshape(-1, 1)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=device).reshape(-1, 1)
    c_obs = torch.tensor(c_obs_np, dtype=torch.float32, device=device).reshape(-1, 1)



    pinn = PINN()
    kernel_net = KernelNet() if learn_kernel else FixedExponentialKernel()

    trainer = MemoryPINNTrainer(pinn, kernel_net, D=D, lr=lr, device=device,
                                 learn_kernel=learn_kernel)



    history = trainer.train(
        x_grid, t_grid, x_ic, c0_ic, t_bc, x_obs, t_obs, c_obs,
        epochs=epochs, print_every=print_every
    )


    Xg, Tg = np.meshgrid(x_np, t_np, indexing="ij")
    x_flat = torch.tensor(Xg.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
    t_flat = torch.tensor(Tg.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)

    with torch.no_grad():
        c_pred_flat = pinn(x_flat, t_flat).cpu().numpy().reshape(Xg.shape)

    err_mse = mse(c_pred_flat, c_np)
    err_rel = relative_l2_error(c_pred_flat, c_np)

    with torch.no_grad():
        t_kernel_eval = torch.linspace(0, T, 200, device=device).reshape(-1, 1)
        K_pred = kernel_net(t_kernel_eval).cpu().numpy().flatten()
    K_true = TRUE_KERNEL_FN(t_kernel_eval.cpu().numpy().flatten())

    plot_heatmaps(x_np, t_np, c_np, c_pred_flat,
                  save_path=os.path.join(out_dir, f"{name}_heatmaps.png"))
    plot_kernel(t_kernel_eval.cpu().numpy().flatten(), K_pred, K_true,
                save_path=os.path.join(out_dir, f"{name}_kernel.png"))
    plot_training_curves(history,
                          save_path=os.path.join(out_dir, f"{name}_loss.png"))

    results = {
        "name": name,
        "mse": err_mse,
        "relative_l2": err_rel,
        "kernel_rel_l2": relative_l2_error(K_pred, K_true),
        "history": history,
    }
    print(f"[{name}] c(x,t) MSE={err_mse:.3e} | rel L2={err_rel:.3e} "
          f"| kernel rel L2={results['kernel_rel_l2']:.3e}")

    return results, pinn, kernel_net


if __name__ == "__main__":
    run_experiment("smoke_test", learn_kernel=True, obs_fraction=0.2,
                    epochs=200, Nx=21, Nt=21, print_every=50)
