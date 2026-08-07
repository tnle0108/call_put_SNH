"""End-to-end run on a synthetic term sheet -- no external data needed.

The bond is shaped like the ones this package was written for: the coupon resets
every 12 months but pays every 6, it carries a floor, and it is callable on dates
that do not fall on coupon dates.

Run with ``python -m callput.examples.demo`` from the project root.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np

from callput import (
    BondSchedule,
    CallPutTree,
    CouponPeriod,
    CurveLeg,
    YieldCurve,
    compile_bond,
)

VALUATION = date(2026, 8, 6)
MATURITY = date(2031, 9, 15)

FACE = 100.0
MARGIN = 0.0120
FLOOR = 0.0550
CURRENT_FIXED = 0.0640  # set at the last reset, before the valuation date
CALL_STRIKE = 1.0

TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
DISCOUNT = [0.0295, 0.0310, 0.0340, 0.0380, 0.0410, 0.0450, 0.0480, 0.0500]
DEPOSIT_12M = [0.0410, 0.0425, 0.0455, 0.0495, 0.0525, 0.0565, 0.0595, 0.0615]

A_R, SIG_R = 0.05, 0.0100
A_L, SIG_L = 0.08, 0.0120
RHO = 0.40

# -- grid ---------------------------------------------------------------------
# Target span in days between filler nodes.  Contract dates are always nodes; this
# only sets how finely the gaps between them are subdivided.  It is a modelling
# choice worth one to two basis points of face on Delta-EVE, so it is declared
# here rather than defaulted anywhere -- `compile_bond` has no default for it.
# See "Choosing the grid" in callput/README.md for the measured convergence.
STEP_DAYS = 21

# Shortest filler the subdivision may create.  Purely a performance guard: `dx` is
# rebuilt per step, so a short step is well conditioned, merely wasteful.
MIN_STEP_DAYS = 3


def days(d: date) -> int:
    return (d - VALUATION).days


def coupon_dates() -> list[date]:
    """15 March and 15 September, from the first one after the valuation date."""
    out = []
    for year in range(VALUATION.year, MATURITY.year + 1):
        for month in (3, 9):
            d = date(year, month, 15)
            if VALUATION < d <= MATURITY:
                out.append(d)
    return sorted(out)


def call_dates() -> list[date]:
    """15 June each year from 2028 -- deliberately off the coupon calendar, which
    is what forces accrued interest into the exercise test."""
    return [date(y, 6, 15) for y in range(2028, MATURITY.year + 1)]


def build(reading: str) -> BondSchedule:
    """``"advance"``: the reset governs the two payments that follow it.

    ``"arrears"``: the reset governs the payment falling due that same day and the
    next one.  Both are internally consistent and both put two payments under one
    fixing; they price differently, and only the prospectus settles which applies.
    """
    pays = coupon_dates()
    starts = [date(2026, 3, 15)] + pays[:-1]
    resets = [d for d in pays if d.month == 9]

    periods = []
    for pay, start in zip(pays, starts):
        accrual = (pay - start).days / 365.0
        if pay <= resets[0]:
            # Rate set at the last reset before the valuation date: already known.
            periods.append(
                CouponPeriod(
                    pay_day=days(pay),
                    accrual=accrual,
                    accrual_start_day=days(start),
                    fixed_rate=CURRENT_FIXED,
                )
            )
            continue
        if reading == "advance":
            fixing = max(r for r in resets if r < pay)
        else:
            fixing = max(r for r in resets if r <= pay)
        periods.append(
            CouponPeriod(
                pay_day=days(pay),
                accrual=accrual,
                accrual_start_day=days(start),
                fixing_day=days(fixing),
                ref_tenor_days=365,
                margin=MARGIN,
                floor=FLOOR,
            )
        )
    return BondSchedule(
        face=FACE,
        maturity_day=days(MATURITY),
        periods=periods,
        call={days(d): CALL_STRIKE for d in call_dates()},
    )


def legs(disc_bump: float = 0.0, ref_bump: float = 0.0):
    disc = YieldCurve.from_zero_rates(TENORS, DISCOUNT).shifted(disc_bump)
    ref = YieldCurve.from_zero_rates(TENORS, DEPOSIT_12M).shifted(ref_bump)
    return (
        CurveLeg(disc, a=A_R, sigma=SIG_R, dcc="ACT/365"),
        CurveLeg(ref, a=A_L, sigma=SIG_L, dcc="ACT/365"),
    )


def make_tree(sched, step_days=STEP_DAYS, **kw):
    bond = compile_bond(sched, step_days=step_days, min_step=MIN_STEP_DAYS)
    disc, ref = legs(**kw)
    return bond, CallPutTree.multi_curve(bond, disc, ref, rho=RHO)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    sched = build("advance")
    bond, tree = make_tree(sched)

    section("Term sheet")
    print(f"valuation {VALUATION}   maturity {MATURITY}   face {FACE:,.0f}")
    print(f"coupon    semi-annual, reset every 12 months, margin {MARGIN:.2%}")
    print(f"floor     {FLOOR:.2%} on the coupon rate (margin included)")
    print(f"call      {CALL_STRIKE:.2f} of face on {len(call_dates())} dates, "
          f"none on a coupon date")

    section("Grid")
    step = bond.step_days()
    print(f"target {STEP_DAYS} days, floor {MIN_STEP_DAYS} days")
    print(f"{bond.n_steps} steps over {int(bond.days[-1])} days; "
          f"spans {step.min()}-{step.max()} days")
    print(f"fixing groups: {len(bond.groups)}, of which "
          f"{sum(g.needs_axis for g in bond.groups)} need the fixing axis")
    for w in bond.warnings:
        print(f"  warning: {w}")

    section("Diagnostics")
    for key, value in tree.diagnostics().items():
        print(f"{key:24s} {value}")

    section("Decomposition (advance reading)")
    started = time.perf_counter()
    parts = tree.decompose()
    elapsed = time.perf_counter() - started
    for key in ("straight", "floor", "cap", "call", "put", "interaction", "full"):
        print(f"{key:14s} {parts[key]:>12.6f}")
    print(f"\n{elapsed:.1f}s for six valuations")
    print("interaction is the part linear attribution cannot reach: the floor")
    print("makes the bond dearer, so it is called more often.")

    section("The two readings of 'rate determination date'")
    print("Priced without the call, because the arrears reading leaves the coupon")
    print("running at each call date fixed only in the future -- see below.")
    for reading in ("advance", "arrears"):
        sched_r = build(reading)
        sched_r.call = {}
        _, alt = make_tree(sched_r)
        parts = alt.decompose()
        print(f"{reading:8s} straight {parts['straight']:>11.6f}   "
              f"floor {parts['floor']:>9.6f}   full {parts['full']:>11.6f}")
    print("\nThe gap is a term-sheet question, not a modelling one -- check the")
    print("prospectus clause on which payments a determination date governs.")
    try:
        _, alt = make_tree(build("arrears"))
        alt.price()
    except ValueError as exc:
        print(f"\nWith the call schedule the arrears reading is refused:\n  {exc}")

    section("Grid convergence (the floor is the slow one, not the call)")
    print(f"{'step_days':>10} {'straight':>12} {'floor':>10} {'call':>10} {'full':>12}")
    for step_days in (56, 42, 28, 21, 14):
        _, conv = make_tree(sched, step_days=step_days)
        d = conv.decompose()
        print(f"{step_days:>10} {d['straight']:>12.6f} {d['floor']:>10.6f} "
              f"{d['call']:>10.6f} {d['full']:>12.6f}")
    print("\nstraight is grid-independent once the drift is fitted, so all of the")
    print("sensitivity sits in the optionality.  Convergence is not monotone: the")
    print("exercise boundary falls between state levels and shifts as the grid")
    print("changes, which leaves a noise floor of about half a basis point of face.")
    print("21 days and finer agree within that; 28 days sits ~1.3bp outside.")

    section("Parallel shocks")
    print("The same grid is used throughout, but that does not make the bias")
    print("cancel: a shock moves the bond into a different exercise regime, so the")
    print("discretisation error of the two runs differs.  Delta-EVE has to be")
    print("converged in its own right, not inferred from a converged price.")
    base = tree.price()
    print(f"\n{'shock':>10} {'price':>12} {'dEVE':>12}")
    for bump in (-0.02, -0.01, 0.01, 0.02):
        _, shocked = make_tree(sched, disc_bump=bump, ref_bump=bump)
        price = shocked.price()
        print(f"{bump:>+10.0%} {price:>12.6f} {price - base:>+12.6f}")


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()
