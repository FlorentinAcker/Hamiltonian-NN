import scipy
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def hamiltonian(state):
    th1, th2, p1, p2 = state[0], state[1], state[2], state[3]
    d = th1 - th2 
    H_kin = (p1**2 + 2*p2**2 - 2*p1*p2*np.cos(d)) / (2*(2 - np.cos(d)**2))
    H_pot = - 2*np.cos(th1) - np.cos(th2)
    return H_kin + H_pot

def derivatives(state):
    th1, th2, p1, p2 = state[0], state[1], state[2], state[3]
    d = th1 - th2
    c = np.cos(d)
    s = np.sin(d)

    D = 2 - c**2
    N = p1**2 + 2*p2**2 - 2*p1*p2*c

    dth1 = (p1 - p2*c) / D
    dth2 = (2*p2 - p1*c) / D

    common = p1*p2*s / D - N*s*c / D**2
    dp1 = -common - 2*np.sin(th1)
    dp2 =  common - np.sin(th2)

    return np.array([dth1, dth2, dp1, dp2])

def rhs(t, z):
    return derivatives(z)


def solve_one_ic(z0, T):
    t_eval = np.linspace(0, T, 500)
    sol = scipy.integrate.solve_ivp(
        fun=rhs,
        t_span = (0, T),
        y0 = z0,
        method='RK45',
        dense_output=False,
        vectorized=False,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    return sol.y.T

def solve(z, T):
    results = Parallel(n_jobs=8)(
        delayed(solve_one_ic)(z0, T) for z0 in z.T
    )
    return results

def generate_ic(N, low=-np.pi, high=np.pi):
    th1 = np.random.uniform(low, high, N)
    th2 = np.random.uniform(low, high, N)
    p1 = np.zeros(N)
    p2 = np.zeros(N)
    z = np.array([th1, th2, p1, p2])
    return z 

def generate_trajectories(N, low, high, T):
    z = generate_ic(N, low, high)
    return (z, np.array(solve(z, T)))

def generate_dataset(N, low, high, T, k=5):
    z, trajs = generate_trajectories(N, low, high, T)
    y0 = trajs[:, :-k, :].reshape(-1, 4)
    y1 = trajs[:, k:, :].reshape(-1, 4)
    dt = T / 499
    return y0, y1, dt 
