"""Demo with known right answers.

Three synthetic value functions in a 6-D goal space, each with an
analytic feasible volume, so every number the estimator prints can be
checked against ground truth:

* ``ours``     — one ellipsoid.                     vol known exactly.
* ``islands``  — 8 well-separated balls.            vol known exactly,
                                                   8 connected components.
* ``tiny``     — an ellipsoid with rho ~ 1e-6.      the regime where plain
                                                   Monte Carlo sees nothing
                                                   and subset simulation is
                                                   the only option.

The islands case is the reason connectivity exists next to volume: same
axis, same budget, but the feasible set is fragmented into 8 pieces, and a
goal-conditioned policy cannot interpolate between them.

Run:  python examples/demo_synthetic.py
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capvol import BoxSpace, compare, learning_curve  # noqa: E402
from capvol.connectivity import sample_feasible  # noqa: E402
from capvol.curve import capability_vs_threshold  # noqa: E402
from capvol.plots import (  # noqa: E402
    plot_capability_vs_threshold,
    plot_feasible_projection,
    plot_learning_curve,
)
from capvol.report import _fmt, to_json, to_markdown  # noqa: E402

D = 6
HALF = 0.9
BOX_LO, BOX_HI = -HALF, HALF

RADII = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.45])
TINY_SCALE = 0.2142          # rho ~ 1e-6: plain MC sees nothing here
BALL_R = 0.35
BALL_HALF = 0.55


def unit_ball_volume(d: int) -> float:
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)


C_D = unit_ball_volume(D)
VOL_G = (BOX_HI - BOX_LO) ** D
VOL_ELLIPSOID = C_D * float(np.prod(RADII))
N_ISLANDS = 2 ** D          # cube vertices: 64 well-separated balls
VOL_ISLANDS = N_ISLANDS * C_D * BALL_R ** D
VOL_TINY = C_D * float(np.prod(RADII * TINY_SCALE))


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
            d2 = ((X - c) ** 2).sum(axis=1) / radius ** 2
            best = np.minimum(best, d2)
        return best

    return V


def cube_vertices(d: int, half: float) -> np.ndarray:
    return half * np.array(list(itertools.product([-1.0, 1.0], repeat=d)))


def main() -> int:
    rng = np.random.default_rng(7)
    space = BoxSpace(np.full(D, BOX_LO), np.full(D, BOX_HI))

    V_ours = ellipsoid_V(RADII)
    V_islands = balls_V(cube_vertices(D, BALL_HALF), BALL_R)
    V_tiny = ellipsoid_V(RADII * TINY_SCALE)

    truth = {"ours": VOL_ELLIPSOID, "islands": VOL_ISLANDS}

    print("=" * 78)
    print("Analytic reference")
    print("=" * 78)
    print(f"  vol(G)          = {_fmt(VOL_G)}")
    print(f"  vol(ours)       = {_fmt(VOL_ELLIPSOID):<10} rho = {_fmt(VOL_ELLIPSOID / VOL_G)}")
    print(f"  vol(islands)    = {_fmt(VOL_ISLANDS):<10} rho = {_fmt(VOL_ISLANDS / VOL_G)}")
    print(f"  vol(tiny)       = {_fmt(VOL_TINY):<10} rho = {_fmt(VOL_TINY / VOL_G)}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Comparison at V_max = 1.0   (n = 100000)")
    print("=" * 78)
    results = compare(
        {"ours": V_ours, "islands": V_islands},
        space,
        v_max=1.0,
        n=100_000,
        rng=rng,
    )
    for r in results:
        e = r.estimate
        t = truth[r.name]
        err = abs(e.vol - t) / t
        cover = e.vol_ci[0] <= t <= e.vol_ci[1]
        print(f"\n  [{r.name}]  {e.method}")
        print(f"    estimated  vol = {_fmt(e.vol):<10} 95% CI [{_fmt(e.vol_ci[0])}, {_fmt(e.vol_ci[1])}]")
        print(f"    analytic   vol = {_fmt(t):<10} relative error = {err * 100:.1f}%   in CI: {cover}")
        if r.connectivity:
            print(f"    connectivity: {r.connectivity.describe()}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Subset simulation regime: rho ~ 1e-6 (plain MC sees 0 hits)")
    print("=" * 78)
    from capvol import estimate

    t = VOL_TINY
    e_direct = estimate(space, V_tiny, v_max=1.0, n=20_000, rng=rng, method="direct")
    e_tiny = estimate(space, V_tiny, v_max=1.0, n=20_000, rng=rng, method="subset")
    print(f"  analytic          : {_fmt(t)}")
    for tag, e in (("direct-lhs", e_direct), ("subset", e_tiny)):
        err = abs(e.vol - t) / t * 100
        print(f"  {tag:<11} n=20000 : {_fmt(e.vol):<10} err {err:5.1f}%   "
              f"95% CI [{_fmt(e.vol_ci[0])}, {_fmt(e.vol_ci[1])}]  ({e.method})")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Capability vs cost budget")
    print("=" * 78)
    thresholds = [0.25, 0.5, 1.0, 2.0, 4.0]
    curves = {}
    for name, V in (("ours", V_ours), ("islands", V_islands)):
        curves[name] = capability_vs_threshold(space, V, thresholds, n=40_000, rng=rng)
        print(f"  {name:<9} " + "  ".join(_fmt(e.vol) for e in curves[name]))

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("Learning curve: vol(F) ~ x^alpha, true alpha = 0.30")
    print("=" * 78)
    TRUE_ALPHA = 0.30
    xs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    r0 = RADII * 0.3585

    def at_budget(x: float):
        return ellipsoid_V(r0 * (x / xs[0]) ** (TRUE_ALPHA / D))

    curve = learning_curve(
        [(x, at_budget(x)) for x in xs],
        space,
        v_max=1.0,
        n=20_000,
        rng=rng,
    )
    for x, v in zip(curve["x"], curve["vol"]):
        print(f"  x={x:>6.1f}   vol = {_fmt(v)}")
    fit = curve["fit"]
    print(f"\n  fitted alpha = {fit['alpha']:.3f}   (true {TRUE_ALPHA})   R^2 = {fit['r2']:.4f}")

    # ------------------------------------------------------------------
    out_dir = ROOT / "reports"
    fig_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    md = to_markdown(
        results,
        v_max=1.0,
        space_volume=VOL_G,
        title="capvol demo — synthetic check against analytic volumes",
        notes=[
            f"analytic vol(ours) = {_fmt(VOL_ELLIPSOID)}, "
            f"analytic vol(islands) = {_fmt(VOL_ISLANDS)}",
            "same goal space, same budget: `ours` is one island, "
            "`islands` is eight — volume and connectivity answer different questions",
            f"subset simulation recovered rho = {e_tiny.rho:.3e} "
            f"against an analytic {t / VOL_G:.3e} at n = 20000",
            f"learning-curve exponent recovered as {fit['alpha']:.3f} "
            f"against a true {TRUE_ALPHA}",
        ],
    )
    (out_dir / "demo_report.md").write_text(md, encoding="utf-8")
    to_json(
        {
            "results": results,
            "tiny": e_tiny,
            "capability_vs_threshold": {"thresholds": thresholds, "curves": curves},
            "learning_curve": curve,
        },
        out_dir / "demo_report.json",
    )

    sets = {}
    for name, V in (("ours", V_ours), ("islands", V_islands)):
        pts, _ = sample_feasible(space, V, v_max=1.0, n=100_000, rng=rng)
        sets[name] = pts

    try:
        plot_capability_vs_threshold(
            thresholds, curves["ours"],
            fig_dir / "capability_vs_threshold.png",
            title="ours: capability vs cost budget",
        )
        plot_learning_curve(curve, fig_dir / "learning_curve.png",
                            xlabel="training investment x")
        plot_feasible_projection(sets, fig_dir / "feasible_projection.png")
    except ImportError:
        print("  (matplotlib not installed: skipping figures — pip install '.[plot]')")

    print()
    print(f"  report : {out_dir / 'demo_report.md'}")
    print(f"  figures: {fig_dir}")

    # ------------------------------------------------------------------
    ok = True
    for r in results:
        t = truth[r.name]
        if not (r.estimate.vol_ci[0] <= t <= r.estimate.vol_ci[1]):
            print(f"  FAIL: analytic volume outside CI for {r.name}")
            ok = False
    if not (e_tiny.vol_ci[0] <= VOL_TINY <= e_tiny.vol_ci[1]):
        print("  FAIL: subset estimate CI misses the analytic tiny volume")
        ok = False
    if abs(fit["alpha"] - TRUE_ALPHA) > 0.05:
        print("  FAIL: power-law exponent not recovered")
        ok = False
    ours_conn = [r for r in results if r.name == "ours"][0].connectivity
    isl_conn = [r for r in results if r.name == "islands"][0].connectivity
    if ours_conn and ours_conn.largest_ratio < 0.9:
        print("  FAIL: `ours` should be a single component")
        ok = False
    if isl_conn and isl_conn.largest_ratio > 0.3:
        print("  FAIL: `islands` should be fragmented")
        ok = False
    print()
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
