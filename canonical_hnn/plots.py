import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_trajectory(traj, dt=0.01):
    """
    traj : torch.tensor (T, 4) with [theta1, theta2, p1, p2]
    dt   : time step (seconds) for display
    """
    # On se met en CPU et on coupe le graph de gradient
    traj = traj.detach().cpu()          # (T, 4)
    theta1 = traj[:, 0]
    theta2 = traj[:, 1]

    x1 = theta1.sin()
    y1 = -theta1.cos()
    x2 = x1 + theta2.sin()
    y2 = y1 - theta2.cos()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.grid(True)

    line, = ax.plot([], [], 'o-', lw=2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(i):
        thisx = [0.0, float(x1[i]), float(x2[i])]
        thisy = [0.0, float(y1[i]), float(y2[i])]
        line.set_data(thisx, thisy)
        time_text.set_text(f't = {i*dt:.2f}s')
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(traj),
                        init_func=init, interval=dt*1000, blit=True)
    plt.show()
    return ani