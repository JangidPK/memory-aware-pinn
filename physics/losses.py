"""
physics/losses.py

Loss terms calculation
    Loss = L_physics + 10*L_ic + 10*L_bc + L_data
"""

import torch


def physics_loss(R):
    # mean squared loss of residuals, simple:)
    return torch.mean(R ** 2)


def ic_loss(pinn, x_ic, c0_ic):
    """
    Loss due to initial condition. 
    It only checks prediction at t=0, 

    x_ic : (N,1) tensor of x locations at t=0
    c0_ic: (N,1) tensor of the true initial condition c0(x)
    """
    t_zero = torch.zeros_like(x_ic)
    c_pred = pinn(x_ic, t_zero)
    return torch.mean((c_pred - c0_ic) ** 2)


def bc_loss(pinn, t_bc):
    """
    Loss due to boundaries, at x=0, at x=L follows Dirichlet 

    t_bc : (N,1) tensor of time points at which 
    
    to enforce c(0,t)=c(1,t)=0
    """
    # at x = 0
    x_left = torch.zeros_like(t_bc)

    # at x = L
    x_right = torch.ones_like(t_bc)

    # what pinn is giving
    c_left = pinn(x_left, t_bc)
    c_right = pinn(x_right, t_bc)

    # it should be zeroooo
    # whatever value is there, it should be the loss
    return torch.mean(c_left ** 2) + torch.mean(c_right ** 2)


def data_loss(pinn, x_obs, t_obs, c_obs):
    """
    This is simple, loss due to data mismatch

    x_obs, t_obs, c_obs : (N,1) tensors of sparse observations
    """
    c_pred = pinn(x_obs, t_obs)
    return torch.mean((c_pred - c_obs) ** 2)


def total_loss(l_physics, l_ic, l_bc, l_data,
               w_physics=2.0, w_ic=5.0, w_bc=3.0, w_data=2.0):
    """
    Fixed-weight sum of all loss terms
    
    Add them all
    """
    return (w_physics * l_physics
            + w_ic * l_ic
            + w_bc * l_bc
            + w_data * l_data)
