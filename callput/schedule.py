"""Bond schedule and the time grid.

The grid lives in **integer days** from the valuation date and is only converted
to year fractions at the last moment, privately, by each :class:`~callput.leg.CurveLeg`.
Working in integers makes "same date" an exact comparison, makes the guard rails
("at least 3 days", "events 2 days apart") mean what they say, and makes the grid
reproducible bit-for-bit across shock scenarios -- which is what keeps a Delta-EVE
free of discretisation noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def build_day_grid(event_days, step_days: int, min_step: int = 3) -> np.ndarray:
    """Time grid in integer days: event dates, plus fillers spread evenly between.

    ``event_days`` must contain ``0`` (the valuation date) and every date the
    contract cares about -- coupon payments, rate fixings, call/put dates and
    maturity.  An exercise date that does not land on a node is the single largest
    source of error in a Bermudan price, so events are **never moved**.

    Fillers are produced by splitting each gap into near-equal integer pieces
    rather than by overlaying a global calendar ruler.  A ruler and a coupon
    schedule never coincide, so their union leaves a sliver (a 9-day step wedged
    between 30-day steps) at *every* event date; splitting per gap cannot.

    ``step_days`` has no default on purpose -- see :func:`compile_bond`.

    ``min_step`` is purely a performance knob: because ``dx`` is rebuilt per step
    (see :mod:`callput.lattice`), a short step is well-conditioned, merely wasteful.
    """
    ev = sorted({int(d) for d in event_days})
    if not ev:
        raise ValueError("event_days is empty")
    if ev[0] < 0:
        raise ValueError("event_days must not be negative")
    if ev[0] != 0:
        raise ValueError("event_days must contain 0 (the valuation date)")
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    if not 1 <= min_step <= step_days:
        raise ValueError("need 1 <= min_step <= step_days")

    grid = [0]
    for lo, hi in zip(ev[:-1], ev[1:]):
        span = hi - lo
        m = max(1, int(span / step_days + 0.5))
        while m > 1 and span / m < min_step:
            m -= 1
        # Integer round-half-up: the k = m term lands exactly on ``hi``.
        grid.extend(lo + (k * span + m // 2) // m for k in range(1, m + 1))
    out = np.asarray(grid, dtype=int)
    if np.any(np.diff(out) <= 0):  # pragma: no cover - defensive
        raise AssertionError("grid is not strictly increasing")
    return out


@dataclass
class CouponPeriod:
    """One interest period.

    Fixed and floating periods coexist in the same bond: a step-up gives each
    period its own ``fixed_rate``, and a bond that is fixed for two years then
    floats simply has ``fixed_rate`` on the early periods and ``fixing_day`` on
    the later ones.  The period whose rate was already set before the valuation
    date must be declared **fixed** at the announced level.

    Floating rate, with the cap/floor applied to the *coupon* rate -- the margin
    goes in first::

        rate = min(max(L(fixing_day, ref_tenor_days) + margin, floor), cap)

    ``ref_tenor_days`` is the tenor of the index (365 for a 12M index) and is
    independent of ``accrual`` (0.5 for a semi-annual coupon): a bond resetting
    annually but paying twice a year has both.  ``ref_delta`` is the year fraction
    of the index's own *quoting* convention, used only to convert the model's
    discount factor into a simple-compounded rate.
    """

    pay_day: int
    accrual: float
    accrual_start_day: int
    fixed_rate: float | None = None
    fixing_day: int | None = None
    margin: float = 0.0
    ref_tenor_days: int | None = None
    ref_delta: float | None = None
    floor: float | None = None
    cap: float | None = None

    def __post_init__(self) -> None:
        if self.accrual <= 0:
            raise ValueError(f"accrual must be > 0 (pay_day={self.pay_day})")
        if self.accrual_start_day >= self.pay_day:
            raise ValueError(
                f"accrual_start_day must precede pay_day (pay_day={self.pay_day})"
            )
        if (self.fixed_rate is None) == (self.fixing_day is None):
            raise ValueError(
                f"period paying at {self.pay_day} must set exactly one of "
                f"fixed_rate (fixed / step-up leg) or fixing_day (floating leg); "
                f"a rate already set before the valuation date is a fixed period"
            )
        if self.is_float:
            if self.fixing_day < 0:
                raise ValueError(
                    f"fixing_day {self.fixing_day} precedes the valuation date; "
                    f"declare that period fixed at the announced rate instead"
                )
            if self.fixing_day > self.pay_day:
                raise ValueError(
                    f"fixing_day must not follow pay_day (pay_day={self.pay_day})"
                )
            if not self.ref_tenor_days or self.ref_tenor_days <= 0:
                raise ValueError(
                    f"floating period paying at {self.pay_day} needs ref_tenor_days"
                )
            if self.ref_delta is None:
                self.ref_delta = self.ref_tenor_days / 365.0
        else:
            if self.floor is not None or self.cap is not None:
                raise ValueError(
                    f"period paying at {self.pay_day} is fixed, so a cap/floor on "
                    f"the rate has no meaning; drop it or make the period floating"
                )
            if self.margin:
                raise ValueError("margin applies to floating periods only")
        if self.floor is not None and self.cap is not None and self.floor > self.cap:
            raise ValueError(f"floor > cap at pay_day={self.pay_day}")

    @property
    def is_float(self) -> bool:
        return self.fixed_rate is None


@dataclass
class BondSchedule:
    """Face, redemption date, interest periods and the exercise calendar.

    ``call`` and ``put`` map a day offset to a **clean** strike quoted as a
    fraction of face (1.0 = par).  Accrued interest is added by the engine, so an
    exercise date need not fall on a coupon date.
    """

    face: float
    maturity_day: int
    periods: list[CouponPeriod]
    call: dict[int, float] = field(default_factory=dict)
    put: dict[int, float] = field(default_factory=dict)

    @property
    def has_floating(self) -> bool:
        return any(p.is_float for p in self.periods)


@dataclass
class Period:
    """A :class:`CouponPeriod` with grid positions resolved."""

    src: CouponPeriod
    pay_step: int
    fix_step: int | None  # None for a fixed period

    @property
    def is_float(self) -> bool:
        return self.src.is_float

    @property
    def is_deferred(self) -> bool:
        """Fixed at one node but paid at a later one -- needs the fixing axis."""
        return self.fix_step is not None and self.pay_step > self.fix_step


@dataclass
class FixingGroup:
    """Every period governed by one rate fixing.

    A bond resetting every 12 months but paying every 6 months puts two payments
    in each group.  The group is the unit over which the cap/floor is applied --
    once, at ``fix_step`` -- so a single floor stays a single option rather than
    being double counted at each payment date.
    """

    fix_step: int
    periods: list[Period]
    deferred: list[Period]
    last_pay_step: int

    @property
    def needs_axis(self) -> bool:
        return bool(self.deferred)


@dataclass
class CompiledBond:
    """A schedule resolved onto a grid, ready for the lattice."""

    days: np.ndarray
    face: float
    maturity_step: int
    periods: list[Period]
    pay_at: dict[int, list[Period]]
    groups: list[FixingGroup]
    expand_at: dict[int, FixingGroup]
    accrue_at: dict[int, Period]
    call: dict[int, float]
    put: dict[int, float]
    warnings: list[str]

    @property
    def n_steps(self) -> int:
        return int(self.days.size - 1)

    @property
    def has_floating(self) -> bool:
        return any(p.is_float for p in self.periods)

    def step_days(self) -> np.ndarray:
        return np.diff(self.days)


def compile_bond(
    sched: BondSchedule, step_days: int, min_step: int = 3
) -> CompiledBond:
    """Resolve a schedule onto a day grid and group the periods by rate fixing.

    ``step_days`` is the target span between filler nodes, in days, and is
    **required**: it is a modelling choice with a measurable price effect, not an
    implementation detail to be defaulted away.  On a five-year bond resetting
    every 12 months and paying every 6, with a floor and off-coupon call dates,
    moving from a 21-day grid to a monthly one shifts Delta-EVE by roughly 1 to
    1.5 basis points of face -- and the bias does **not** cancel between the base
    and shocked runs, because a shock moves the bond into a different exercise
    regime.

    Measured on that bond, deviations from a 7-day grid in basis points of face::

        step   straight   floor    call    dEVE(-200bp)   dEVE(+200bp)
         56d      +0.26   +2.06   +0.56          -2.11          -1.58
         28d      -0.02   +0.91   -0.23          -1.01          -1.08
         21d      -0.04   -0.35   -0.27          +0.52          +0.54
         14d      -0.09   -0.43   -0.40          +0.09          +0.60

    21 days and finer agree within about 0.6 bp; 28 days sits roughly 1.3 bp
    outside that cluster.  Refining below 14 days buys nothing: the residual is
    the exercise boundary falling between state levels, which oscillates rather
    than converging, so the engine's noise floor is around half a basis point of
    face however fine the grid.  Cost scales near ``dt**-2.5``.

    Note that convergence is not monotone.  Judge a grid by whether it joins the
    fine-grid cluster, not by the gap to the next coarser one, and check it on
    ``decompose()["floor"]`` and on Delta-EVE rather than on the price: the
    discounted cash flows are grid-independent once the drift is fitted, so all of
    the sensitivity lives in the optionality.
    """
    if sched.face <= 0:
        raise ValueError("face must be > 0")
    if sched.maturity_day <= 0:
        raise ValueError("maturity_day must be > 0")
    if not sched.periods:
        raise ValueError("bond has no interest periods")

    events: set[int] = {0, int(sched.maturity_day)}
    for p in sched.periods:
        events.add(int(p.pay_day))
        if p.is_float:
            events.add(int(p.fixing_day))
    events.update(int(d) for d in sched.call)
    events.update(int(d) for d in sched.put)

    late = sorted(d for d in events if d > sched.maturity_day)
    if late:
        print(late)
        raise ValueError(f"event days after maturity: {late}")

    days = build_day_grid(events, step_days=step_days, min_step=min_step)
    index = {int(d): i for i, d in enumerate(days)}

    pay_days = [p.pay_day for p in sched.periods]
    dupes = {d for d in pay_days if pay_days.count(d) > 1}
    if dupes:
        raise ValueError(f"more than one period pays on day(s) {sorted(dupes)}")

    periods = [
        Period(
            src=p,
            pay_step=index[int(p.pay_day)],
            fix_step=None if not p.is_float else index[int(p.fixing_day)],
        )
        for p in sorted(sched.periods, key=lambda q: q.pay_day)
    ]

    pay_at: dict[int, list[Period]] = {}
    for p in periods:
        pay_at.setdefault(p.pay_step, []).append(p)

    groups = _build_groups(periods)
    _check_group_consistency(groups)

    expand_at = {g.last_pay_step: g for g in groups if g.needs_axis}
    if len(expand_at) != sum(1 for g in groups if g.needs_axis):  # pragma: no cover
        raise AssertionError("two fixing groups share a last payment step")

    accrue_at: dict[int, Period] = {}
    for p in periods:
        start, end = p.src.accrual_start_day, p.src.pay_day
        for i, d in enumerate(days):
            if start < d < end:
                if i in accrue_at:
                    raise ValueError(
                        f"interest periods overlap around day {int(d)}: both the "
                        f"period paying at {accrue_at[i].src.pay_day} and the one "
                        f"paying at {end} are accruing"
                    )
                accrue_at[i] = p

    warnings = _collect_warnings(sched, days, step_days, min_step)

    return CompiledBond(
        days=days,
        face=float(sched.face),
        maturity_step=index[int(sched.maturity_day)],
        periods=periods,
        pay_at=pay_at,
        groups=groups,
        expand_at=expand_at,
        accrue_at=accrue_at,
        call={index[int(d)]: float(k) for d, k in sched.call.items()},
        put={index[int(d)]: float(k) for d, k in sched.put.items()},
        warnings=warnings,
    )


def _build_groups(periods: list[Period]) -> list[FixingGroup]:
    by_fix: dict[int, list[Period]] = {}
    for p in periods:
        if p.fix_step is not None:
            by_fix.setdefault(p.fix_step, []).append(p)

    groups = [
        FixingGroup(
            fix_step=fix,
            periods=members,
            deferred=[p for p in members if p.is_deferred],
            last_pay_step=max(p.pay_step for p in members),
        )
        for fix, members in sorted(by_fix.items())
    ]

    # Augmented segments [fix_step, last_pay_step] must tile, not nest: the
    # rollback carries at most one fixing axis at a time.
    for lo, hi in zip(groups[:-1], groups[1:]):
        if lo.last_pay_step > hi.fix_step:
            raise ValueError(
                f"fixing periods overlap: the fixing at step {lo.fix_step} still "
                f"governs a payment at step {lo.last_pay_step}, past the next "
                f"fixing at step {hi.fix_step}"
            )
    return groups


def _check_group_consistency(groups: list[FixingGroup]) -> None:
    """Periods sharing one fixing must share its terms.

    They consume the same fixed rate, so differing margins, caps, floors or index
    tenors within a group describe no valid structure -- it is a term-sheet
    contradiction, and catching it here beats silently pricing one of them.
    """
    fields = ("margin", "ref_tenor_days", "ref_delta", "floor", "cap")
    for g in groups:
        first = g.periods[0].src
        for other in g.periods[1:]:
            differing = [
                f for f in fields if getattr(first, f) != getattr(other.src, f)
            ]
            if differing:
                raise ValueError(
                    f"periods paying at {first.pay_day} and {other.src.pay_day} "
                    f"share the fixing at step {g.fix_step} but disagree on "
                    f"{differing}; one fixing yields one rate"
                )


def _collect_warnings(
    sched: BondSchedule, days: np.ndarray, step_days: int, min_step: int
) -> list[str]:
    out: list[str] = []
    steps = np.diff(days)

    for i in np.flatnonzero(steps < min_step):
        out.append(
            f"step {int(i)} spans {int(steps[i])} day(s) "
            f"({int(days[i])} -> {int(days[i + 1])}): two events sit close "
            f"together; correct but one rollback is nearly wasted"
        )

    # A gap under 1.5 * step_days rounds to a single piece and survives unsplit.
    # The commonest case is valuation date -> first coupon, which leaves the
    # coarsest step in the whole tree sitting right at the root, and it moves with
    # every reporting date.
    for i in np.flatnonzero(steps > 1.25 * step_days):
        out.append(
            f"step {int(i)} spans {int(steps[i])} days against a {step_days}-day "
            f"target ({int(days[i])} -> {int(days[i + 1])}): the gap was too short "
            f"to split, so this is the coarsest step in the tree; a slightly "
            f"smaller step_days would halve it"
        )

    # A missing period prices without complaint, so say something.  Reported
    # rather than raised because an unusual but deliberate schedule should not be
    # blocked; promote to an error if your bonds are always contiguous.
    ordered = sorted(sched.periods, key=lambda p: p.pay_day)
    for earlier, later in zip(ordered[:-1], ordered[1:]):
        if later.accrual_start_day != earlier.pay_day:
            out.append(
                f"interest schedule is not contiguous: the period paying on day "
                f"{earlier.pay_day} ends there but the next one starts accruing on "
                f"day {later.accrual_start_day}; a missing period would price "
                f"silently"
            )
    if ordered[-1].pay_day != int(sched.maturity_day):
        out.append(
            f"the last coupon pays on day {ordered[-1].pay_day} but the bond "
            f"redeems on day {int(sched.maturity_day)}; the final period is "
            f"usually paid together with the principal"
        )

    for d in sorted(set(sched.call) & set(sched.put)):
        out.append(f"day {int(d)} carries both a call and a put strike")
    if int(sched.maturity_day) in set(sched.call) | set(sched.put):
        out.append("an exercise date on the maturity date has no economic effect")
    return out
