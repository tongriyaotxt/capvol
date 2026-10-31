# capvol

**Measure what a goal-conditioned policy can actually reach — as a number, with a confidence interval.**

[中文文档](README.zh-CN.md)

---

Success rate is the number everyone quotes and nobody can compare. It depends on the task set you chose, the reset distribution you sampled, and how many retries you allowed. Two labs reporting "92%" are not measuring the same thing, so *"this robot is better than that one"* stays unfalsifiable.

If you have a goal-conditioned value function `V(s, g)`, the set of things the robot can do has a definition:

```
F(s0) = { g ∈ G : V(s0, g) < V_max }
```

and the quantity everyone wants but almost nobody computes:

```
capability = vol(F)
```

`capvol` estimates that volume for an arbitrary `V`, in the regime where naive Monte Carlo returns zero, and reports an interval instead of a point.

It also measures the thing volume cannot see: **whether the feasible set is one island or a dust of fragments.** A policy that reaches 500 unrelated goals cannot interpolate between them, so every new goal is a new skill.

---

## Install

```bash
git clone https://github.com/tongriyaotxt/capvol.git
cd capvol
pip install -e ".[plot,test]"
```

Only `numpy` is required. `matplotlib` is used by the demo and the plot helpers; `pytest` for the test suite.

---

## Quickstart

```python
import numpy as np
from capvol import BoxSpace, estimate

# 6-D goal space: end-effector pose envelope, say
space = BoxSpace(low=-0.9 * np.ones(6), high=0.9 * np.ones(6))

# your value function: V(X) with X of shape [N, d], returns [N]
# (wrap a single-point callable with capvol.pointwise)
def V(X):
    r = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45])
    return ((np.atleast_2d(X) / r) ** 2).sum(axis=1)

e = estimate(space, V, v_max=1.0, n=100_000)
print(e.describe())
# direct-lhs: rho=1.034e-02 [9.71e-03, 1.10e-02]  vol=3.517e-01  n=100000  ok=1034
```

That is the whole interface. Everything else is a view on top of it.

---

## What you can measure

| Function | Question it answers |
|---|---|
| `estimate(space, V, v_max)` | How much of the goal space can this policy reach? |
| `connectivity(points)` | Is the reachable set one island, or many? |
| `capability_vs_threshold(space, V, thresholds)` | How does capability grow as I allow more cost? |
| `learning_curve(checkpoints, space, v_max)` | Does capability scale with training, and with what exponent? |
| `compare({name: V}, space, v_max)` | Which of these methods reaches more, on the same axes? |
| `to_markdown(results)` / `to_json(...)` | A table a reviewer can check |

---

## How the estimator works

Two regimes, one API. The regime is picked automatically, and the result always says which one ran.

### Direct — feasible volume is a visible fraction of `G`

Latin-hypercube sampling in the goal space, count the feasible samples, report a **Wilson** interval. LHS stratifies every coordinate, so it removes most of the variance plain Monte Carlo leaves behind at the same sample count. Wilson (not the normal approximation) is what keeps the interval honest when the count is small.

### Subset simulation — feasible volume is tiny

A 6-D goal space where the robot reaches a thin slice has `rho ~ 1e-6`. Direct sampling needs a million draws to see one hit. Subset simulation (Au & Beck, 2001) writes the probability as a product of conditional ones and estimates every factor with the same `n` samples:

```
P(F) = p0^k · P(F | F_k),      F_j = { V < t_j }
```

**The ladder approaches `V_max` from above**, `t_1 > t_2 > ... > t_k > V_max`. This direction is the whole trick: if you stop at the *first* threshold above `V_max` you get plain Monte Carlo wearing a costume, and if the thresholds cross below `V_max` the decomposition no longer holds. `capvol` descends while the next threshold stays above `V_max`, then measures the last factor by counting.

**Mixing matters more than it looks.** Conditional samples come from random-walk Metropolis chains seeded at the survivors. One step per sample leaves the population clustered around the seeds — which are the *lowest-cost* points — so the final fraction is measured on a population biased toward the feasible set. Measured against an analytic ellipsoid in 6-D, the bias is:

| steps per chain | relative error (rho ≈ 1e-3) | relative error (rho ≈ 2e-5) |
|---|---|---|
| 1 | +17% | +23% |
| 3 | +4.5% | +1.1% |
| **10 (default)** | **+1.8%** | **+0.7%** |
| 30 | +0.5% | +2.0% |

Ten steps is the default. It costs `10 n` value calls per level — that is the honest price of an unbiased answer.

### When nothing is found

If the sampler sees zero feasible points, the result is **not** `rho = 0`. It is a detection limit:

```
rho_ci = (0, 3/n)          # rule of three, 95% upper bound
```

A zero would be a claim; a bound is a measurement.

---

## Validation

The demo has closed-form answers for everything, so no number in this section is a promise.

```bash
python examples/demo_synthetic.py
```

```
Analytic reference
  vol(G)          = 34.01
  vol(ours)       = 0.3516     rho = 0.01034
  vol(islands)    = 0.608      rho = 0.01788
  vol(tiny)       = 3.396e-05  rho = 9.985e-07

Comparison at V_max = 1.0   (n = 100000)

  [ours]  direct-lhs
    estimated  vol = 0.3326     95% CI [0.3125, 0.354]
    analytic   vol = 0.3516     relative error = 5.4%   in CI: True
    connectivity: components=1  largest=1.000

  [islands]  direct-lhs
    estimated  vol = 0.6136     95% CI [0.5861, 0.6423]
    analytic   vol = 0.608      relative error = 0.9%   in CI: True
    connectivity: components=62  largest=0.027

Subset simulation regime: rho ~ 1e-6 (plain MC sees 0 hits)
  analytic          : 3.396e-05
  direct-lhs  : 0            err 100.0%   95% CI [0, 6.532e-03]
  subset      : 3.485e-05    err   2.6%   95% CI [3.141e-05, 3.865e-05]

Learning curve: vol(F) ~ x^alpha, true alpha = 0.30
  fitted alpha = 0.294   (true 0.3)   R^2 = 0.9924

  RESULT: PASS
```

What each case proves:

| Case | What it checks |
|---|---|
| `ours` — one ellipsoid | direct estimator is unbiased and its interval covers truth |
| `islands` — 64 disjoint balls | volume is right **and** the set is reported as fragmented (62 components, largest 2.7%) |
| `tiny` — `rho ≈ 1e-6` | direct sampling returns a bound; subset returns the true value within 3% |
| learning curve | the power-law fit recovers a known exponent (0.294 vs 0.30) |

Test suite: **19 tests, 4.2 s**, every estimator checked against a closed-form answer.

```bash
python -m pytest tests -q
```

---

## Reading the numbers

Four rules. Ignoring any of them makes the comparison meaningless.

1. **`vol(F)` is meaningless without `V_max`.** Capability is a function of how much cost you tolerate. Always report the pair. `capability_vs_threshold` is there so you can pick a defensible `V_max` instead of a convenient one.

2. **Two volumes computed in different goal spaces are different quantities.** Same space, same budget, or the table is fiction. `compare` enforces this by taking one space for all methods.

3. **Volume and connectivity answer different questions.** In the demo, `islands` has 1.7× the volume of `ours` and 62 disconnected components. If you only report volume, you rank the fragmented policy first.

4. **`n_ok` is part of the result.** An estimate built on 20 hits and one built on 20,000 are not the same claim, even when both say "±5%". The estimator string and `n_ok` are in every report for exactly this reason.

---

## API

```python
# goal spaces
BoxSpace(low, high)                  # axis-aligned
DataSpace(points, jitter=0.1)        # goals around observed states

# the one function
estimate(space, V, v_max=1.0, n=20000, method="auto", rng=None) -> VolumeEstimate

# adapters
pointwise(f)                         # g -> float  ==>  X -> [N]

# connectivity
connectivity(points, k=8, scale=1.0) -> ConnectivityReport
sample_feasible(space, V, v_max, n, min_points=600) -> (points, estimate)

# curves
capability_vs_threshold(space, V, thresholds, n) -> list[VolumeEstimate]
learning_curve([(x, V_x), ...], space, v_max) -> dict      # includes .fit.alpha
power_law_fit(x, y) -> dict

# comparison
compare({"ours": V_a, "baseline": V_b}, space, v_max, n) -> list[MethodResult]
to_markdown(results, path=...) / to_json(obj, path=...)
```

`V` takes `X` of shape `[N, d]` and returns `[N]`. Non-finite values are treated as unreachable — that is how you express "the model is out of its depth here" without special-casing.

---

## Connectivity: why not k-NN

A k-NN graph is the obvious choice and it is wrong here. With sparse samples, each point's 10 nearest neighbours can all live in *other* blobs, so a k-NN graph silently merges a fragmented set into one component.

`capvol` uses a **mutual k-NN graph**: each point gets a local radius `r_i` = distance to its own `k`-th neighbour, and two points are linked only if

```
dist(i, j) <= min(r_i, r_j)
```

The radius follows local density for free, so it survives a goal space where density varies by orders of magnitude, and the `min` makes the test conservative — a dense point never drags a sparse neighbour across a gap. The failure mode is over-splitting, which is the safe direction: it never fabricates connectivity that is not there.

The cost is sample hunger: you need roughly ≥ 20 feasible points per expected blob to resolve it. `sample_feasible` keeps drawing until `min_points` is reached for exactly this reason, independent of how many samples the volume estimate used.

---

## Limitations

- **It measures `V`, not the robot.** If `V` is wrong, `vol(F)` is a precise measurement of a wrong number. Validate `V` first (rank correlation against outcomes is the cheapest check).
- **Subset simulation needs a bootstrap.** With `rho·n < ~3` the first threshold cannot be placed and the estimator reports a detection limit. Pass a larger `n`, or use gradient-informed sampling — there is no way around it.
- **Connectivity is empirical.** It is a statement about the samples you drew at the resolution you drew them, not a topological proof.
- **Dimension hurts twice.** Both the volume and the number of samples needed scale badly; the design target is `d ≤ 10`, which covers end-effector poses, mobile bases, and quadrupeds, and is not enough for a 40-DoF whole-body goal space without decomposition.
- **Cost is `mcmc_steps × n` value calls per level.** For a neural `V` that is real compute. Batch it.

---

## Intended use

`capvol` was written for the SuperMimic-α line of work, where the claim under test is that a goal-conditioned value function makes capability a measurable geometric quantity instead of an adjective. It applies to any system that has an explicit `V(s, g)`: goal-conditioned RL, MPC with a terminal cost, or a learned reachability critic.

If you have `V` and want to know whether your system is actually getting more capable, this is the measurement. If you do not have `V`, there is nothing here for you yet — and that is the point.

---

## License

MIT. See [LICENSE](LICENSE).
