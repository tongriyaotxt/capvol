"""The two curves worth putting in a paper.

``capability_vs_threshold``
    How much of the goal space is reachable as you relax the cost budget.
    A flat-left curve means "reach a little or reach nothing"; a long
    tail means graded capability.  Always report ``V_max`` with a number
    taken from this curve, not one picked for taste.

``learning_curve``
    ``vol(F)`` against training investment, plus a power-law fit.
    The exponent is the thing a scaling law is made of:

        vol(F) ~ x ** alpha

    with ``alpha`` the marginal return on data / GPU-hours / rollouts.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from .estimator import VolumeEstimate, estimate

__all__ = ["capability_vs_threshold", "learning_curve", "power_law_fit"]


def capability_vs_threshold(
    space,
    V,
    thresholds: Sequence[float],
    n: int = 20000,
    method: str = "auto",
    rng: np.random.Generator | None = None,
) -> list[VolumeEstimate]:
    """Estimate ``vol(F)`` for every ``V_max`` in ``thresholds``."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = []
    for vm in thresholds:
        out.append(estimate(space, V, v_max=float(vm), n=n, method=method, rng=rng))
    return out


def power_law_fit(x: Sequence[float], y: Sequence[float]) -> dict:
    """Least-squares fit of ``y = c * x**alpha`` in log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {"alpha": float("nan"), "c": float("nan"), "r2": float("nan"),
                "n_points": int(x.size)}

    lx, ly = np.log(x), np.log(y)
    alpha, intercept = np.polyfit(lx, ly, 1)
    pred = alpha * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "alpha": float(alpha),
        "c": float(np.exp(intercept)),
        "r2": float(r2),
        "n_points": int(x.size),
    }


def learning_curve(
    checkpoints: Iterable[tuple[float, Callable]],
    space,
    v_max: float = 1.0,
    n: int = 20000,
    method: str = "auto",
    rng: np.random.Generator | None = None,
) -> dict:
    """``vol(F)`` against training investment.

    Parameters
    ----------
    checkpoints :
        ``[(investment, V_at_that_checkpoint), ...]``.  Investment can be
        hours, GPU-hours, rollouts, episodes — anything monotone.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    xs, vols, lo, hi = [], [], [], []
    for x, V in checkpoints:
        e = estimate(space, V, v_max=v_max, n=n, method=method, rng=rng)
        xs.append(float(x))
        vols.append(e.vol)
        lo.append(e.vol_ci[0])
        hi.append(e.vol_ci[1])

    fit = power_law_fit(xs, vols)
    return {
        "x": xs,
        "vol": vols,
        "vol_lo": lo,
        "vol_hi": hi,
        "fit": fit,
        "v_max": v_max,
    }
