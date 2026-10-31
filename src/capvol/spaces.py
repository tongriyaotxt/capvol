"""Goal spaces.

Every space is sampled through the unit cube [0,1]^d.  Working in the unit
cube keeps the subset-simulation MCMC uniform: proposals are clipped by
reflection, so the conditional distribution stays uniform inside the box.

Only two concrete spaces are provided on purpose:

* ``BoxSpace``   — axis-aligned box, e.g. an end-effector pose envelope.
* ``DataSpace``  — a box spanned by observed goal states plus Gaussian
                   jitter around each observation (goals drawn from a
                   human motion library, for instance).

Both expose the same four methods, so the estimator never needs to know
which one it is looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GoalSpace", "BoxSpace", "DataSpace", "lhs_unit", "random_unit"]


def random_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Plain uniform sampling in the unit cube."""
    return rng.random((n, d))


def lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sampling in the unit cube.

    Each of the ``d`` coordinates is stratified into ``n`` equal bins and
    the strata are randomly permuted per coordinate.  For a smooth
    indicator this removes most of the variance plain Monte Carlo leaves
    behind, at zero extra cost.
    """
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


class GoalSpace:
    """Interface every goal space must satisfy."""

    dim: int

    def volume(self) -> float:
        raise NotImplementedError

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_unit(self, n: int, rng: np.random.Generator, mode: str = "lhs") -> np.ndarray:
        if mode == "random":
            return random_unit(n, self.dim, rng)
        if mode == "lhs":
            return lhs_unit(n, self.dim, rng)
        raise ValueError(f"unknown sampling mode {mode!r}")

    def sample(self, n: int, rng: np.random.Generator, mode: str = "lhs") -> np.ndarray:
        return self.from_unit(self.sample_unit(n, rng, mode))


@dataclass
class BoxSpace(GoalSpace):
    """Axis-aligned box ``[low, high]``."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=float).ravel()
        self.high = np.asarray(self.high, dtype=float).ravel()
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same length")
        if not np.all(self.high > self.low):
            raise ValueError("every high[i] must exceed low[i]")

    @property
    def dim(self) -> int:
        return int(self.low.size)

    def volume(self) -> float:
        return float(np.prod(self.high - self.low))

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return (X - self.low) / (self.high - self.low)

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return self.low + U * (self.high - self.low)

    @classmethod
    def from_points(cls, points: np.ndarray, pad: float = 0.05) -> "BoxSpace":
        P = np.atleast_2d(np.asarray(points, dtype=float))
        lo, hi = P.min(axis=0), P.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        return cls(lo - pad * span, hi + pad * span)


class DataSpace(GoalSpace):
    """Goals drawn around observed states.

    ``volume()`` is the bounding-box volume of the observations — the same
    measure the box estimator would use, so numbers stay comparable
    between the two spaces.
    """

    def __init__(self, points: np.ndarray, jitter: float = 0.1, pad: float = 0.0):
        P = np.atleast_2d(np.asarray(points, dtype=float))
        if P.shape[0] < 2:
            raise ValueError("need at least two observation points")
        self.points = P
        self.jitter = float(jitter)
        self.box = BoxSpace.from_points(P, pad=pad)
        # per-dimension scale so jitter is dimensionless
        self._std = P.std(axis=0)
        self._std = np.where(self._std > 1e-12, self._std, 1.0)

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    def volume(self) -> float:
        return self.box.volume()

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.box.to_unit(X), 0.0, 1.0)

    def from_unit(self, U: np.ndarray) -> np.ndarray:
        return self.box.from_unit(U)

    def sample_unit(self, n: int, rng: np.random.Generator, mode: str = "lhs") -> np.ndarray:
        idx = rng.integers(0, self.points.shape[0], size=n)
        base = self.points[idx]
        noise = rng.normal(scale=self.jitter * self._std, size=(n, self.dim))
        return self.to_unit(base + noise)
