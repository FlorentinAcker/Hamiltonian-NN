#%%
import torch

# %%
def hamiltonian(state):
    """Takes as argument a state and returns the Hamiltonian"""

def derivatives(state):
    th1, th2, p1, p2 = state.unbind(-1)
    d = th1 - th2 
    det = 2 - d.cos()**2 
    N = p1**2 + p2**2 - 2*p1*p2*d.cos()
    dth1 = (p1 - p2*d.cos()) / det
    dth2 = (2*p2 - p1*d.cos()) / det    
    dp1 = p1*p2*d.sin()/det - N*d.sin()*d.cos()/det**2 - 2*th1.sin()
    dp2= p1*p2*d.sin()/det + N*d.sin()*d.cos()/det**2 - th2.sin()

def iter_midpoint(state, dt, nb_iter=5):
    z_mid = state.clone()
    for i in range(nb_iter):
        z_mid = state + dt/2*derivatives(z_mid)
    return z_mid

def step_midpoint(state, dt, nb_iter=5):
    z_mid = iter_midpoint(state, dt, nb_iter=nb_iter)
    return state + dt*derivatives(z_mid)

def simulate_batch(z0, dt, T, nb_iter=5):
    N_steps = int(T / dt)
    states = torch.zeros(z0.shape[0], N_steps+1, 4)
    states[0, :, 0] = z0 
    for i in range(1, N_steps + 1):
        states[:, i, :] = step_midpoint(states[:, i-1, :], dt=dt, nb_iter=nb_iter)
        return states




# %%
