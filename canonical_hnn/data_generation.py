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
