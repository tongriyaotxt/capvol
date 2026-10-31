"""Plots. matplotlib is optional — importing this module never fails."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .estimator import VolumeEstimate

__all__ = [
    "plot_capability_vs_threshold",
    "plot_learning_curve",
    "plot_feasible_projection",
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_capability_vs_threshold(
    thresholds: Sequence[float],
    estimates: Sequence[VolumeEstimate],
    path: str | Path | None = None,
    title: str = "Capability vs cost budget",
) -> str | None:
    plt = _plt()
    v = np.array([e.vol for e in estimates])
    lo = np.array([e.vol_ci[0] for e in estimates])
    hi = np.array([e.vol_ci[1] for e in estimates])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(thresholds, v, "o-", color="#1f77b4", label="vol(F)")
    ax.fill_between(thresholds, lo, hi, color="#1f77b4", alpha=0.2, lw=0, label="95% CI")
    ax.set_xlabel(r"$V_{max}$  (cost budget)")
    ax.set_ylabel(r"$\mathrm{vol}(F)$")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    plt.close(fig)
    return None


def plot_learning_curve(
    curve: dict,
    path: str | Path | None = None,
    title: str = "Capability vs training investment",
    xlabel: str = "training investment",
) -> str | None:
    plt = _plt()
    x = np.array(curve["x"])
    y = np.array(curve["vol"])
    lo = np.array(curve["vol_lo"])
    hi = np.array(curve["vol_hi"])
    fit = curve.get("fit", {})

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(x, y, "o", color="#d62728", label="measured")
    ax.fill_between(x, lo, hi, color="#d62728", alpha=0.15, lw=0)
    if np.isfinite(fit.get("alpha", np.nan)):
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, fit["c"] * xs ** fit["alpha"], "--", color="k",
                label=fr"fit: $\alpha$={fit['alpha']:.2f} ($R^2$={fit['r2']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{vol}(F)$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    plt.close(fig)
    return None


def plot_feasible_projection(
    sets: dict,
    path: str | Path | None = None,
    title: str = "Feasible set (shared 2-D projection)",
) -> str | None:
    """Project every method's feasible samples onto one shared SVD basis."""
    plt = _plt()
    pooled = np.vstack([np.atleast_2d(p) for p in sets.values()])
    if pooled.shape[0] < 3:
        return None
    centre = pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(pooled - centre, full_matrices=False)
    basis = vt[:2].T

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (name, P) in enumerate(sets.items()):
        P = np.atleast_2d(P)
        if P.shape[0] == 0:
            continue
        Y = (P - centre) @ basis
        ax.scatter(Y[:, 0], Y[:, 1], s=6, alpha=0.5,
                   color=colors[i % len(colors)], label=f"{name} (n={P.shape[0]})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    plt.close(fig)
    return None
