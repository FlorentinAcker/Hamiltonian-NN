import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_generation import solve_one_ic, generate_ic

def animate_trajectory(traj, dt=0.01):
    """
    traj : np.ndarray (T, 4) with [theta1, theta2, p1, p2]
    dt   : time step (seconds) for display
    """
    theta1 = traj[:, 0]
    theta2 = traj[:, 1]

    x1 =  np.sin(theta1)
    y1 = -np.cos(theta1)
    x2 = x1 + np.sin(theta2)
    y2 = y1 - np.cos(theta2)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.grid(True)

    line,      = ax.plot([], [], 'o-', lw=2)
    time_text  = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(i):
        thisx = [0.0, x1[i], x2[i]]
        thisy = [0.0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(f't = {i*dt:.2f}s')
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(traj),
                        init_func=init, interval=dt*1000, blit=True)
    return ani   # pas de plt.show() ici