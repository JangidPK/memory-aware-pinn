"""
training/train_baseline.py

an ordinary PINN for dc/dt = D * c_xx, verified against an FD
solution. This is thebasic step before tackling the
memory-integral version.

"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_ordinary_diffusion, gaussian_pulse
from models.pinn import PINN
from training.trainer import BaselineTrainer
from utils.metrics import mse, relative_l2_error
from utils.plotting import plot_heatmaps, plot_training_curves

torch.manual_seed(0)
np.random.seed(0)


def run_baseline(D=0.01, Nx=41, Nt=41, L=1.0, T=1.0, n_colloc=2000,
                  epochs=3000, lr=1e-3, device="cpu", out_dir="results",
                  print_every=500):
    os.makedirs(out_dir, exist_ok=True)

    # ground truth
    x_np, t_np, c_np = generate_ordinary_diffusion(D=D, Nx=Nx, Nt=Nt, L=L, T=T)




    # data points 
    rng = np.random.default_rng(0)
    x_colloc = torch.tensor(rng.uniform(0, L, n_colloc), dtype=torch.float32).reshape(-1, 1)
    t_colloc = torch.tensor(rng.uniform(0, T, n_colloc), dtype=torch.float32).reshape(-1, 1)



    # initial conditions point
    x_ic = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1)
    c0_ic = torch.tensor(gaussian_pulse(x_np), dtype=torch.float32).reshape(-1, 1)
    c0_ic[0] = 0.0
    c0_ic[-1] = 0.0



    # boundary conditions points 
    t_bc = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1)

    # train 
    pinn = PINN()
    trainer = BaselineTrainer(pinn, D=D, lr=lr, device=device)
    history = trainer.train(x_colloc, t_colloc, x_ic, c0_ic, t_bc,
                             epochs=epochs, print_every=print_every)

    # evaluate 
    Xg, Tg = np.meshgrid(x_np, t_np, indexing="ij")
    x_flat = torch.tensor(Xg.flatten(), dtype=torch.float32).reshape(-1, 1)
    t_flat = torch.tensor(Tg.flatten(), dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        c_pred = pinn(x_flat, t_flat).numpy().reshape(Xg.shape)

    err_mse = mse(c_pred, c_np)
    err_rel = relative_l2_error(c_pred, c_np)
    print(f"[Baseline] MSE={err_mse:.3e} | relative L2={err_rel:.3e}")

    plot_heatmaps(x_np, t_np, c_np, c_pred,
                  save_path=os.path.join(out_dir, "baseline_heatmaps.png"))
    plot_training_curves(history,
                          save_path=os.path.join(out_dir, "baseline_loss.png"))

    return {"mse": err_mse, "relative_l2": err_rel, "history": history}, pinn


if __name__ == "__main__":
    run_baseline(epochs=500, print_every=100)
