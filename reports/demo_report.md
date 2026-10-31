# capvol demo — synthetic check against analytic volumes

## Setup

- goal-space volume `vol(G)` = 34.01
- cost budget `V_max` = 1
- samples per estimate: 100000

## Results

| method | vol(F) | 95% CI | rho | estimator | components | largest ratio |
|---|---|---|---|---|---|---|
| ours | 0.3326 | [0.3125, 0.354] | 9.780e-03 | direct-lhs | 1 | 1.000 |
| islands | 0.6136 | [0.5861, 0.6423] | 0.01804 | direct-lhs | 62 | 0.027 |

## Relative to `ours`

| method | vol ratio | delta |
|---|---|---|
| islands | 1.845x | 0.2809 |

## Notes

- analytic vol(ours) = 0.3516, analytic vol(islands) = 0.608
- same goal space, same budget: `ours` is one island, `islands` is eight — volume and connectivity answer different questions
- subset simulation recovered rho = 1.025e-06 against an analytic 9.985e-07 at n = 20000
- learning-curve exponent recovered as 0.294 against a true 0.3

---

`vol(F)` is measured, not claimed: `F = {g : V(g) < V_max}` and the estimate comes with a 95% interval. Compare rows only inside this table — a volume computed in another goal space at another budget is a different quantity.