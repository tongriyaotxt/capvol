"""capvol — measure what a goal-conditioned policy can actually reach.

The core object is the feasible set

    F(s0) = { g in G : V(s0, g) < V_max }

and the quantity everyone quotes but nobody can compute:

    capability = vol(F)

This package estimates vol(F) for an arbitrary value function V, with a
confidence interval, for goal spaces where naive Monte Carlo is hopeless
(high dimension, small feasible volume).
"""

from .spaces import BoxSpace, DataSpace, GoalSpace, lhs_unit, random_unit
from .estimator import VolumeEstimate, estimate, pointwise, subset_simulation
from .connectivity import ConnectivityReport, connectivity
from .curve import learning_curve, capability_vs_threshold
from .compare import MethodResult, compare
from .report import to_json, to_markdown

__all__ = [
    "BoxSpace",
    "DataSpace",
    "GoalSpace",
    "lhs_unit",
    "random_unit",
    "VolumeEstimate",
    "estimate",
    "pointwise",
    "subset_simulation",
    "ConnectivityReport",
    "connectivity",
    "learning_curve",
    "capability_vs_threshold",
    "MethodResult",
    "compare",
    "to_json",
    "to_markdown",
]

__version__ = "0.1.0"
