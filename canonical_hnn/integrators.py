import torch 
import numpy as np
from data_generation import hamiltonian
from scipy.integrate import solve_ivp


def energy_along_traj(traj):
    return hamiltonian(traj.T).T

def learned_energy_along_traj(model, traj_np):
    model.eval()
    traj_t = torch.tensor(traj_np, dtype=torch.float32)
    with torch.no_grad():
        return model.forward(traj_t).squeeze(-1).numpy()
    
def integrate_rk45(model, y0, T, max_step=0.01):
    model.eval()
    if hasattr(y0, 'numpy'):
        y0 = y0.numpy()

    def f_theta(t, y):
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        y_t.requires_grad_(True)
        H = model.forward(y_t).squeeze(-1)
        gradH = torch.autograd.grad(H.sum(), y_t, create_graph=False)[0]
        dydt = torch.einsum('ij, aj -> ai', model.J.T, gradH)
        return dydt.detach().squeeze(0).numpy()

    sol = solve_ivp(f_theta, [0, T], y0, method='RK45',
                    max_step=max_step, dense_output=False)
    return sol.y.T