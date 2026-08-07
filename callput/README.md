# `callput`

Hull-White pricing for VCB IRRBB bonds with embedded call, put, cap and floor.
Self-contained: it does not import from `src/`, which stays in place as the
reference implementation the cross-check tests measure against.

## Which engine

| | `CallPutTree.single_curve` | `CallPutTree.multi_curve` |
|---|---|---|
| Bonds | Fixed rate, **step-up included** | Floating, **wholly or partly** |
| Factors | 1 (discounting) | 2 (discounting + coupon index), correlated by `rho` |
| Curves | 1 | 2, each with its own day-count convention |
| Embedded options | Call / put | Call / put **+ cap / floor on the coupon rate** |

Neither structure needs special handling: every interest period is an independent
record, so a step-up is just a different `fixed_rate` per period, and a bond fixed
for its first two years is `fixed_rate` on the early periods and `fixing_day` on
the later ones. A fixed period cannot carry a cap or floor — there is no rate
optionality once the rate is set — and `compile_bond` says so rather than pricing
it.

## Quick start

```python
from datetime import date
from callput import BondSchedule, CallPutTree, CouponPeriod, CurveLeg, YieldCurve, compile_bond

VALUATION = date(2026, 8, 6)
days = lambda d: (d - VALUATION).days          # everything is integer day offsets

sched = BondSchedule(
    face=100.0,
    maturity_day=days(date(2031, 9, 15)),
    periods=[
        CouponPeriod(                          # rate set before valuation: fixed
            pay_day=days(date(2026, 9, 15)),
            accrual_start_day=days(date(2026, 3, 15)),
            accrual=184 / 365,
            fixed_rate=0.064,
        ),
        CouponPeriod(                          # reset 12M, pays 6M -> deferred
            pay_day=days(date(2027, 3, 15)),
            accrual_start_day=days(date(2026, 9, 15)),
            accrual=181 / 365,
            fixing_day=days(date(2026, 9, 15)),
            ref_tenor_days=365,                # 12M index, 6M accrual
            margin=0.012,
            floor=0.055,
        ),
        # ...
    ],
    call={days(date(2028, 6, 15)): 1.0},       # clean strike, fraction of face
)

bond = compile_bond(sched, step_days=21)      # required -- see "Choosing the grid"
tree = CallPutTree.multi_curve(
    bond,
    CurveLeg(disc_curve, a=0.05, sigma=0.010, dcc="ACT/365"),
    CurveLeg(deposit_12m_curve, a=0.08, sigma=0.012, dcc="ACT/365"),
    rho=0.40,
)

tree.price()          # dirty PV
tree.decompose()      # straight / floor / cap / call / put / interaction / full
tree.diagnostics()    # curve repricing error, realised correlation, grid stats
```

`python -m callput.examples.demo` runs the whole thing on a synthetic term sheet.

## What the design commits to

**The grid lives in integer days.** Coupon, reset, fixing and exercise dates are
grid nodes and are never moved; fillers are produced by splitting each gap evenly
towards `step_days`, not by overlaying a calendar ruler (a ruler never coincides
with a coupon schedule, so their union leaves a sliver at every event date).
Integers make "same date" exact, make the guard rails mean what they say, and make
the grid reproducible bit-for-bit — which is what keeps a shocked run's ΔEVE free
of discretisation noise. The grid never depends on rates, so every Basel scenario
and every key-rate bump reprices on the same lattice.

**A reset that governs several payments is carried as a label.** With a 12-month
reset paying every six months, one fixing sets several coupons, and by the time a
backward rollback reaches a payment date the recombining lattice has forgotten
which node the fixing came from. Between the last payment it governs and the
fixing date, the value array grows a leading axis with one copy per possible fixed
level. That axis does not branch and carries no probability of its own; the copies
differ only in the cash injected and in the exercise decisions they consequently
take. At the fixing date the label becomes fact — it is that node's own level — and
the axis collapses along its diagonal.

**Caps and floors bound the coupon rate, margin included:**

```
rate = min(max(L + margin, floor), cap)
```

applied **once per fixing**, never once per payment. A 12-month reset paying twice
a year observes its floor once and both coupons inherit the result; applying it per
payment date would price one contractual floor as two options.

**Each leg owns its day-count convention** and the two need not agree. The curve
part of the affine bond price is a ratio of discount factors — a pure number, the
year fraction cancels — so only `B(tau)` and `G(tau)` consume a clock, and those
are exactly the terms answering "how much uncertainty accumulates here".
Conventions that are not linear in actual days (30/360, ACT/ACT) are refused as a
diffusion clock: they are for counting money, not time. `a` and `sigma` must be
calibrated in their own leg's clock; `CurveLeg.retimed` converts when they were
not.

**Exercise dates need not fall on coupon dates.** Strikes are clean, quoted as a
fraction of face, and accrued interest is added by the engine using the
capped/floored rate — getting that wrong shifts the exercise boundary exactly in
the scenarios where the floor is biting.

**Spacing is rebuilt per step** (`dx_i = sqrt(3 V_i)`), which keeps the branch
probabilities in Hull's standard form however uneven the grid is. The state space
is truncated at one half-width in `x` (`n_sigma` terminal standard deviations,
capped by Hull's band), from which each level's count follows — truncating per
step instead makes the level count jump between adjacent times and clips a whole
band of nodes at once.

**The drift is fitted to the curve.** Affine node rates alone reproduce the market
curve only to a few tenths of a basis point of face; that residual is small but
systematic and does not cancel between a base and a shocked run. One Arrow-Debreu
pass and a scalar per step remove it, so `diagnostics()["curve_repricing_error"]`
comes back at machine precision.

## Reading the decomposition

```
full = straight + floor - cap - call + put + interaction
```

`interaction` is what linear attribution cannot reach. For a bond that is both
floored and callable it is not noise: the floor makes the bond dearer and
therefore likelier to be called, so the features eat into each other. It is
reported on its own rather than spread silently over the components.

## Choosing the grid

`step_days` has **no default**. It is a modelling choice with a measurable price
effect, not an implementation detail, and burying it in a default hides a decision
worth one to two basis points of face.

Measured on `examples/demo.py` — a five-year bond resetting every 12 months,
paying every 6, floored, callable off the coupon calendar — as deviations from a
7-day grid, in basis points of face:

| step | straight | floor | call | ΔEVE −200bp | ΔEVE +200bp | sec / decompose |
|---:|---:|---:|---:|---:|---:|---:|
| 56d | +0.26 | +2.06 | +0.56 | −2.11 | −1.58 | 0.5 |
| 42d | +0.04 | +1.79 | −0.36 | — | — | 0.9 |
| 28d | −0.02 | +0.91 | −0.23 | −1.01 | −1.08 | 2.5 |
| **21d** | −0.04 | −0.35 | −0.27 | +0.52 | +0.54 | 9.7 |
| 14d | −0.09 | −0.43 | −0.40 | +0.09 | +0.60 | 41.3 |
| 7d | 0 | 0 | 0 | 0 | 0 | 260.8 |

Four things this says:

**`straight` is grid-independent.** Fitting the drift to the curve removes it as a
source of error, so all of the remaining sensitivity lives in the optionality.
Check convergence on `decompose()["floor"]` and on ΔEVE, not on the price.

**The floor converges more slowly than the call.** It is also the component that
changes sign between 28 and 21 days — that jump, about 1.3 bp, is the real
boundary.

**The bias does not cancel in ΔEVE.** At 28 days the price is 1.26 bp high while
ΔEVE is about 1.05 bp low. Using the same grid for base and shocked runs removes
discretisation noise from the discounting but not from the optionality: a shock
moves the bond into a different exercise regime. ΔEVE has to be converged in its
own right.

**Convergence is not monotone.** Exercise dates sit exactly on nodes by
construction, but the exercise boundary in *state* space still falls between
levels and shifts as `dx` changes. That leaves a noise floor of roughly half a
basis point of face that refining does not remove — 14d and 7d differ by about as
much as 21d and 14d do. Judge a grid by whether it joins the fine-grid cluster,
not by the gap to the next coarser one.

So: **21 days** joins the cluster at 4× the cost of a monthly grid and 1/27th of a
weekly one. A monthly grid is defensible if 1 to 1.5 bp of face is inside your
materiality threshold, but then say so with the number attached. A weekly grid
buys nothing. If you ever need to go below the noise floor, average two nearby
grids (20 and 22 days) rather than refining — that targets the oscillation
directly, at twice the cost instead of twenty-seven times.

`compile_bond` warns when a gap was too short to split and left a step materially
longer than the target. The usual culprit is valuation date to first coupon, which
puts the coarsest step of the whole tree at the root and moves with every
reporting date.

Cost scales near `dt**-2.5`. `n_fix` trades accuracy for speed by carrying a
coarse label grid and interpolating at the collapse — around a basis point of face
at 21 labels, so the exact grid stays the default.

## Limits

- **Fixing conventions must come from the prospectus.** Which payments a rate
  determination date governs is a term-sheet question and the readings price
  differently; `compile_bond` takes the mapping as data and rejects contradictions
  (a payment claimed by two fixings, fixing periods that nest) rather than
  guessing. `examples/demo.py` prices two readings side by side.
- **An in-arrears period exercised mid-period is refused.** Its accrued interest
  depends on a rate not yet fixed; the contract has to supply a rule and this
  engine does not invent one.
- **An index averaged over an observation window** (common for "average 12M
  deposit rate of the four state-owned banks") is Asian-style path dependence and
  is not modelled.
- **Call notice periods** are not modelled: pass the decision date as the exercise
  date.
- **No calibration.** `a`, `sigma` and `rho` are inputs.

## Tests

```
pytest tests -q
```

The two that carry the most weight are cross-checks rather than tolerances:
`test_crosscheck.py` prices against `src/multi_hw_tree.py` on a uniform grid and
watches the gap shrink as the grid refines, and `test_multi_curve.py` prices the
fixing axis against an independent in-advance collapse that values each deferred
coupon at its fixing node. A binding floor turns a floater into a fixed-rate bond,
so the two engines must also agree exactly — which pins the cap/floor convention
at the same time.
