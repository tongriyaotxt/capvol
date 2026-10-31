"""Is the feasible set one island, or a dust of fragments?

Volume alone cannot tell "the robot can do a lot of things" from "the
robot can do a lot of unrelated things".  The second is much less useful:
a policy whose feasible set is 400 disconnected blobs cannot interpolate
between them, so every new goal is a new skill.

**How the graph is built.**  A fixed-radius or plain k-NN graph is the
obvious choice and both are wrong here.  A fixed radius cannot follow a
density that varies by orders of magnitude across a 6-D goal space.  A
plain k-NN graph is worse: with sparse samples each point's ``k`` nearest
neighbours can all live in *other* blobs, so it silently merges
everything.

So we use a **mutual k-NN graph**: point ``i`` gets a local radius
``r_i`` = distance to its own ``k``-th neighbour, and two points are
linked only if they are within *both* radii,

    link(i, j)  <=>  dist(i, j) <= min(r_i, r_j)

The radius adapts to local density for free, and the ``min`` makes the
test symmetric-conservative: a dense point never drags a sparse
neighbour across a gap.  The failure mode is over-splitting (a thin
bridge between two dense regions can be cut), which is the safe
direction for this metric — it never fabricates connectivity that is not
there.

    connected_ratio = |largest component| / |feasible samples|

1.0 means the feasible set is (empirically) one piece.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ConnectivityReport", "connectivity", "sample_feasible"]


@dataclass
class ConnectivityReport:
    n_points: int
    n_components: int
    largest_ratio: float
    link_radius: float      # median local radius, normalised space
    k: int

    def describe(self) -> str:
        return (
            f"components={self.n_components}  largest={self.largest_ratio:.3f}  "
            f"n={self.n_points}  k={self.k}  r_med={self.link_radius:.4f}"
        )


def _union_find(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def connectivity(
    points: np.ndarray,
    k: int = 8,
    scale: float = 1.0,
    max_points: int = 2000,
    rng: np.random.Generator | None = None,
) -> ConnectivityReport:
    """Connected-component analysis of the feasible set.

    Parameters
    ----------
    points :
        Feasible goals, shape ``[N, d]``.  Usually ``sample_feasible(...)``.
    k :
        Neighbourhood size for the local radius.  ``k=8`` works well for
        sets that hold at least ~20 samples per expected blob.  Raising it
        merges near-touching blobs; lowering it splits thin bridges.
    scale :
        Multiplier on the local radius.  ``1.0`` is the conservative
        default; ``1.2-1.5`` is useful when you want bridges preserved.
    max_points :
        Subsample above this size — O(N^2) distances, so 2000 is the
        practical ceiling.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    P = np.atleast_2d(np.asarray(points, dtype=float))
    n = P.shape[0]
    if n < 4:
        return ConnectivityReport(n, n, 1.0 if n else 0.0, 0.0, k)

    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        P = P[idx]
        n = max_points

    lo = P.min(axis=0)
    span = P.max(axis=0) - lo
    diag = float(np.linalg.norm(span)) or 1.0
    Pn = (P - lo) / diag

    diff = Pn[:, None, :] - Pn[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)

    kk = int(min(max(k, 1), n - 1))
    kth = np.partition(d, kk - 1, axis=1)[:, kk - 1]
    radius = np.minimum(kth[:, None], kth[None, :]) * scale

    mask = np.triu(d <= radius, k=1)
    rows, cols = np.where(mask)
    pairs = np.stack([rows, cols], axis=1) if rows.size else np.zeros((0, 2), int)

    roots = _union_find(n, pairs)
    _, counts = np.unique(roots, return_counts=True)

    return ConnectivityReport(
        n_points=n,
        n_components=int(counts.size),
        largest_ratio=float(counts.max() / n),
        link_radius=float(np.median(kth)),
        k=kk,
    )


def sample_feasible(
    space,
    V,
    v_max: float,
    n: int = 20000,
    rng: np.random.Generator | None = None,
    method: str = "auto",
    min_points: int = 600,
    max_batches: int = 60,
    batch: int = 50000,
) -> tuple[np.ndarray, object]:
    """Draw feasible goals, whatever regime they live in.

    Connectivity needs *coverage*, not just an estimate: a handful of
    points cannot resolve a fragmented set.  So this keeps drawing until
    ``min_points`` feasible points are in hand (or the batch budget runs
    out), independent of how many samples the volume estimate used.

    Returns ``(points, estimate)``.
    """
    from .estimator import _clean, estimate

    if rng is None:
        rng = np.random.default_rng(0)

    est = estimate(space, V, v_max=v_max, n=n, method=method, rng=rng,
                   keep_samples=True)

    chunks: list[np.ndarray] = []
    if "samples" in est.extra:
        chunks.append(space.from_unit(est.extra["samples"]))
    collected = sum(c.shape[0] for c in chunks)

    tries = 0
    while collected < min_points and tries < max_batches:
        U = space.sample_unit(batch, rng, mode="random")
        X = space.from_unit(U)
        v = _clean(V(X), X.shape[0])
        ok = X[v < v_max]
        if ok.shape[0]:
            chunks.append(ok)
            collected += ok.shape[0]
        tries += 1

    if not chunks:
        return np.zeros((0, space.dim)), est
    return np.vstack(chunks), est
