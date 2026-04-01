#%%
%matplotlib inline
import torch
import scipy
#%%
def hamiltonian(state):
    th1, th2, p1, p2 = state.unbind(-1)
    d = th1 - th2 
    H_kin = (p1**2 + 2*p2**2 - 2*p1*p2*d.cos()) / (2*(2 - d.cos()**2))
    H_pot = - 2*th1.cos() - th2.cos()
    return H_kin + H_pot

def derivatives(state):
    th1, th2, p1, p2 = state.unbind(-1)
    d = th1 - th2
    c = d.cos()
    s = d.sin()

    D = 2 - c**2
    N = p1**2 + 2*p2**2 - 2*p1*p2*c

    dth1 = (p1 - p2*c) / D
    dth2 = (2*p2 - p1*c) / D

    common = p1*p2*s / D - N*s*c / D**2
    dp1 = -common - 2*th1.sin()
    dp2 =  common - th2.sin()

    return torch.stack((dth1, dth2, dp1, dp2), dim=-1)

