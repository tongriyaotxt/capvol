"""Compare methods on the same axes.

Two methods are comparable only if they are measured in the same goal
space with the same ``V_max``.  This module enforces exactly that and
returns one row per method, so the output is a table and not an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .connectivity import ConnectivityReport, connectivity, sample_feasible
from .estimator import VolumeEstimate, estimate

__all__ = ["MethodResult", "compare"]


@dataclass
class MethodResult:
    name: str
    estimate: VolumeEstimate
    connectivity: ConnectivityReport | None = None

    def row(self) -> dict:
        r = {
            "method": self.name,
            "vol": self.estimate.vol,
            "vol_lo": self.estimate.vol_ci[0],
            "vol_hi": self.estimate.vol_ci[1],
            "rho": self.estimate.rho,
            "n_ok": self.estimate.n_ok,
            "estimator": self.estimate.method,
        }
        if self.connectivity is not None:
            r["components"] = self.connectivity.n_components
            r["largest_ratio"] = self.connectivity.largest_ratio
        return r


def compare(
    methods: Mapping[str, Callable],
    space,
    v_max: float = 1.0,
    n: int = 20000,
    method: str = "auto",
    with_connectivity: bool = True,
    rng: np.random.Generator | None = None,
) -> list[MethodResult]:
    """Evaluate several value functions in one shared goal space.

    Parameters
    ----------
    methods :
        ``{"ours": V_ours, "tracking": V_tracking}``.
    space :
        The single goal space every method is scored in.
    v_max :
        The shared cost budget.  Changing it changes the ranking, so
        never compare two numbers measured at two different budgets.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    out: list[MethodResult] = []
    for name, V in methods.items():
        est = estimate(space, V, v_max=v_max, n=n, method=method, rng=rng)
        conn = None
        if with_connectivity:
            pts, _ = sample_feasible(space, V, v_max=v_max, n=n, method=method, rng=rng)
            conn = connectivity(pts, rng=rng)
        out.append(MethodResult(name=name, estimate=est, connectivity=conn))
    return out
