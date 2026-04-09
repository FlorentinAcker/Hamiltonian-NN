from typing import Dict, Iterable, Mapping, Sequence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trajectory(
    traj: np.ndarray,
    T: float | None = None,
    dt: float | None = None,
    max_frames: int = 300,
    figsize: tuple[float, float] = (2.8, 2.8),
) -> FuncAnimation:
    """
    traj : np.ndarray (n_steps, 4) with [theta1, theta2, p1, p2]
    T    : total physical time
    dt   : physical time step between states, if known
    max_frames : hard cap to keep the embedded HTML small
    figsize : Matplotlib figure size
    """
    n_steps: int = len(traj)

    if dt is None and T is not None:
        dt = float(T) / max(n_steps - 1, 1)
    elif dt is None and T is None:
        dt = 1.0
    else:
        dt = float(dt)

    n_frames: int = min(max_frames, n_steps)
    idx: np.ndarray = np.linspace(0, n_steps - 1, n_frames).astype(int)

    theta1 = traj[idx, 0]
    theta2 = traj[idx, 1]

    x1 = np.sin(theta1)
    y1 = -np.cos(theta1)
    x2 = x1 + np.sin(theta2)
    y2 = y1 - np.cos(theta2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.grid(True)

    line, = ax.plot([], [], "o-", lw=1.8)
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=8)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(i: int):
        thisx = [0.0, float(x1[i]), float(x2[i])]
        thisy = [0.0, float(y1[i]), float(y2[i])]
        line.set_data(thisx, thisy)
        time_text.set_text(f"t = {float(idx[i]) * dt:.2f}s")
        return line, time_text

    if len(idx) > 1:
        mean_diff: float = float(np.mean(np.diff(idx)))
        frame_dt: float = mean_diff * dt
    else:
        frame_dt = dt

    interval_ms: float = 1000.0 * frame_dt

    ani = FuncAnimation(
        fig,
        update,
        frames=len(idx),
        init_func=init,
        interval=interval_ms,
        blit=True,
    )
    plt.close(fig)
    return ani


COLORS: Dict[str, str] = {
    "royal_blue": "#2B4EAE",
    "forest": "#2D6A4F",
    "mustard": "#C8930A",
    "purple": "#6B2D8B",
    "crimson": "#A4243B",
    "slate": "#4A5568",
    "black": "#1A1A1A",
}
PALETTE: list[str] = list(COLORS.values())


def set_academic_style() -> None:
    """Apply an academic Matplotlib style (grid, serif fonts, sober colors)."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            "grid.color": "#AAAAAA",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "lines.linewidth": 1.5,
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "legend.frameon": True,
        }
    )


def plot_loss(
    hists: Mapping[str, dict | Sequence[float]],
    title: str = "Loss curves",
    figsize: tuple[float, float] = (8.0, 3.5),
) -> None:
    """
    hists: dict {scheme_name: {"train": [...], "test": [...]} or list}
    If a scheme only provides a list (not a dict), it is assumed to be train-only.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for (name, hist), color in zip(hists.items(), PALETTE, strict=False):
        if isinstance(hist, dict):
            ax.plot(hist["train"], color=color, lw=1.5, label=f"{name} — train")
            ax.plot(hist["test"], color=color, lw=1.0, ls="--", label=f"{name} — test")
        else:
            ax.plot(hist, color=color, lw=1.5, label=name)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_conservation(
    trajs: Mapping[str, np.ndarray],
    t_axes: Mapping[str, np.ndarray],
    traj_ref: np.ndarray,
    t_ref: np.ndarray,
    title: str = "True energy conservation",
    figsize: tuple[float, float] = (8.0, 3.5),
) -> None:
    """
    trajs   : {scheme_name: array (n, 4)} — HNN trajectories
    t_axes  : {scheme_name: array (n,)}   — associated time axes
    traj_ref: array (n, 4) — reference trajectory
    t_ref   : array (n,)   — reference time axis
    """
    from integrators import energy_along_traj

    E_ref = energy_along_traj(traj_ref)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        t_ref,
        E_ref - E_ref[0],
        color=COLORS["black"],
        lw=1.2,
        ls="--",
        label="RK45 (reference)",
    )

    for (name, traj), color in zip(trajs.items(), PALETTE, strict=False):
        E = energy_along_traj(traj)
        ax.plot(t_axes[name], E - E[0], color=color, label=name)

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$H(y(t)) - H(y(0))$")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_comparison(
    traj_hnn: np.ndarray,
    t_hnn: np.ndarray,
    traj_ref: np.ndarray,
    t_ref: np.ndarray,
    model,
    scheme_name: str = "SHNN",
    figsize: tuple[float, float] = (9.0, 3.5),
) -> None:
    """Compare true H and learned \hat{H}_θ along one HNN trajectory."""
    from integrators import energy_along_traj, learned_energy_along_traj

    E_true = energy_along_traj(traj_hnn)
    E_hat = learned_energy_along_traj(model, traj_hnn)
    E_ref = energy_along_traj(traj_ref)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        t_ref,
        E_ref - E_ref[0],
        color=COLORS["black"],
        lw=1.2,
        ls="--",
        label="RK45 + $H$ (true)",
    )
    ax.plot(
        t_hnn,
        E_true - E_true[0],
        color=COLORS["royal_blue"],
        lw=1.2,
        label=f"{scheme_name} — $H$ (true)",
    )
    ax.plot(
        t_hnn,
        E_hat - E_hat[0],
        color=COLORS["crimson"],
        lw=1.0,
        ls=":",
        label=f"{scheme_name} — $\\hat{{H}}_\\theta$",
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$H(y(t)) - H(y(0))$")
    ax.set_title(f"Energy conservation — {scheme_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_full(
    t_hnn: np.ndarray,
    t_ref: np.ndarray,
    E_hnn: np.ndarray,
    E_ref: np.ndarray,
    E_hat: np.ndarray,
    scheme_name: str = "HNN",
    figsize: tuple[float, float] = (9.0, 4.0),
) -> None:
    """
    Plot true H and learned \hat{H}_θ on both trajectories (HNN + reference),
    using already computed energy arrays.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        t_ref,
        E_ref - E_ref[0],
        color=COLORS["black"],
        lw=1.2,
        ls="--",
        label="RK45 — $H$ (true, reference)",
    )
    ax.plot(
        t_hnn,
        E_hnn - E_hnn[0],
        color=COLORS["royal_blue"],
        lw=1.5,
        label=f"{scheme_name} — $H$ (true)",
    )
    ax.plot(
        t_hnn,
        E_hat - E_hat[0],
        color=COLORS["crimson"],
        lw=1.0,
        ls=":",
        label=f"{scheme_name} — $\\hat{{H}}_\\theta$",
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$H(y(t)) - H(y(0))$")
    ax.set_title(f"Energy conservation — {scheme_name} vs reference")
    ax.legend()
    plt.tight_layout()
    plt.show()