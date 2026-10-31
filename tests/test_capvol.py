"""Tests. The point is not coverage, it is that every estimator has a
closed-form answer to be checked against."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from capvol import BoxSpace, DataSpace, compare, estimate, pointwise
from capvol.connectivity import connectivity, sample_feasible
from capvol.curve import learning_curve, power_law_fit
from capvol.estimator import subset_simulation, wilson_interval
from capvol.spaces import lhs_unit

D = 6
HALF = 0.9


def unit_ball_volume(d: int) -> float:
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)


C6 = unit_ball_volume(D)


def ellipsoid_V(radii):
    r = np.asarray(radii, dtype=float)

    def V(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return ((X / r) ** 2).sum(axis=1)

    return V


def balls_V(centres, radius):
    C = np.atleast_2d(np.asarray(centres, dtype=float))

    def V(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        best = np.full(X.shape[0], np.inf)
        for c in C:
            best = np.minimum(best, ((X - c) ** 2).sum(axis=1) / radius ** 2)
        return best

    return V


@pytest.fixture
def space():
    return BoxSpace(np.full(D, -HALF), np.full(D, HALF))


# ---------------------------------------------------------------- spaces
def test_box_roundtrip(space):
    X = np.random.default_rng(0).uniform(-HALF, HALF, size=(7, D))
    np.testing.assert_allclose(space.from_unit(space.to_unit(X)), X, atol=1e-12)


def test_box_volume(space):
    assert space.volume() == pytest.approx((2 * HALF) ** D)


def test_lhs_is_stratified():
    u = lhs_unit(64, 3, np.random.default_rng(0))
    for j in range(3):
        bins = np.floor(u[:, j] * 64).astype(int)
        assert len(set(bins.tolist())) == 64      # one sample per stratum


def test_data_space_roundtrip():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3))
    sp = DataSpace(pts, jitter=0.05)
    assert sp.dim == 3
    assert sp.volume() > 0
    X = sp.sample(100, rng)
    assert X.shape == (100, 3)


# ------------------------------------------------------------ statistics
def test_wilson_bounds():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo, hi = wilson_interval(100, 100)
    assert 0.95 < lo <= 1.0 and hi >= 0.999


# ------------------------------------------------------- volume: direct
def test_direct_ellipsoid(space):
    radii = np.array([0.85, 0.75, 0.65, 0.55, 0.50, 0.45])
    truth = C6 * float(np.prod(radii))
    e = estimate(space, ellipsoid_V(radii), v_max=1.0, n=200_000,
                 method="direct", rng=np.random.default_rng(1))
    assert e.method == "direct-lhs"
    assert abs(e.vol - truth) / truth < 0.10
    assert e.vol_ci[0] <= truth <= e.vol_ci[1]


# ------------------------------------------------------- volume: subset
@pytest.mark.parametrize("scale", [0.6775, 0.3585, 0.2142])
def test_subset_matches_analytic(space, scale):
    """Three orders of magnitude of rho, one estimator."""
    radii = scale * np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.45])
    truth = C6 * float(np.prod(radii))
    e = subset_simulation(space, ellipsoid_V(radii), v_max=1.0, n=20_000,
                          rng=np.random.default_rng(2))
    assert abs(e.vol - truth) / truth < 0.20, e.describe()
    assert e.vol_ci[0] <= truth <= e.vol_ci[1], e.describe()
    assert e.extra["n_levels"] >= 2


def test_subset_beats_direct_in_the_tiny_regime(space):
    """Direct MC sees nothing here; that must be reported as a bound."""
    radii = 0.2142 * np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.45])
    V = ellipsoid_V(radii)
    truth = C6 * float(np.prod(radii))

    d = estimate(space, V, v_max=1.0, n=20_000, method="direct",
                 rng=np.random.default_rng(3))
    assert d.rho == 0.0
    assert d.vol_ci[1] > 0.0                     # a bound, not a zero

    s = subset_simulation(space, V, v_max=1.0, n=20_000,
                          rng=np.random.default_rng(3))
    assert abs(s.vol - truth) / truth < 0.20
    # direct returns a detection limit (3/n), subset returns an interval
    # two orders of magnitude tighter
    assert s.vol_ci[1] < d.vol_ci[1] / 10


def test_auto_picks_the_right_regime(space):
    big = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45])
    small = 0.2142 * big
    rng = np.random.default_rng(4)
    e1 = estimate(space, ellipsoid_V(big), v_max=1.0, n=50_000, method="auto", rng=rng)
    e2 = estimate(space, ellipsoid_V(small), v_max=1.0, n=50_000, method="auto", rng=rng)
    assert e1.method.startswith("direct")
    assert e2.method.startswith("subset")


def test_non_finite_is_unreachable(space):
    radii = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

    def V(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = ((X / radii) ** 2).sum(axis=1)
        out[X[:, 0] > 0.2] = np.nan          # half the space returns NaN
        return out

    e = estimate(space, V, v_max=1.0, n=20_000, method="direct",
                 rng=np.random.default_rng(5))
    assert np.isfinite(e.vol)
    assert e.vol < C6 * float(np.prod(radii))


def test_pointwise_adapter(space):
    radii = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45])
    batch = ellipsoid_V(radii)
    single = lambda g: float(batch(np.asarray(g)[None, :])[0])
    wrapped = pointwise(single)
    X = np.random.default_rng(0).uniform(-1, 1, size=(11, D))
    np.testing.assert_allclose(wrapped(X), batch(X))


# -------------------------------------------------------- connectivity
def test_connectivity_single_blob():
    rng = np.random.default_rng(0)
    P = rng.normal(scale=0.3, size=(800, D))
    rep = connectivity(P, k=8, rng=rng)
    assert rep.largest_ratio > 0.85


def test_connectivity_detects_islands():
    rng = np.random.default_rng(0)
    centres = 0.55 * np.array(list(itertools.product([-1.0, 1.0], repeat=D)))
    blobs = [c + rng.normal(scale=0.08, size=(60, D)) for c in centres]
    P = np.vstack(blobs)
    rep = connectivity(P, k=8, rng=rng)
    assert rep.n_components >= 40
    assert rep.largest_ratio < 0.1


def test_sample_feasible_reaches_min_points(space):
    radii = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45])
    pts, est = sample_feasible(space, ellipsoid_V(radii), v_max=1.0,
                               n=50_000, min_points=500,
                               rng=np.random.default_rng(6))
    assert pts.shape[0] >= 500
    assert pts.shape[1] == D


# --------------------------------------------------------------- curves
def test_power_law_fit_exact():
    x = np.array([1.0, 10.0, 100.0])
    y = 3.0 * x ** 0.42
    fit = power_law_fit(x, y)
    assert fit["alpha"] == pytest.approx(0.42, abs=1e-8)
    assert fit["c"] == pytest.approx(3.0, rel=1e-8)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-8)


def test_learning_curve_recovers_exponent(space):
    true_alpha, d = 0.30, D
    xs = [1.0, 4.0, 16.0, 64.0]
    base = 0.3585 * np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.45])
    cps = [(x, ellipsoid_V(base * (x ** (true_alpha / d)))) for x in xs]
    curve = learning_curve(cps, space, v_max=1.0, n=20_000,
                           rng=np.random.default_rng(7))
    assert abs(curve["fit"]["alpha"] - true_alpha) < 0.08
    assert curve["fit"]["r2"] > 0.9


# -------------------------------------------------------------- compare
def test_compare_rows_are_comparable(space):
    ours = ellipsoid_V(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45]))
    islands = balls_V(0.55 * np.array(list(itertools.product([-1.0, 1.0], repeat=D))), 0.35)
    res = compare({"ours": ours, "islands": islands}, space, v_max=1.0,
                  n=50_000, rng=np.random.default_rng(8))
    assert [r.name for r in res] == ["ours", "islands"]
    assert all(r.estimate.space_volume == pytest.approx(space.volume()) for r in res)
    assert res[0].connectivity.largest_ratio > 0.85
    assert res[1].connectivity.largest_ratio < 0.2
