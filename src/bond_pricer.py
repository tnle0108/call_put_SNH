"""
One term-sheet row + one valuation date -> CallPutTree.

The core property is that the module is **pure in ``rpd``**. It reads no files, writes no files and
reads no globals; everything goes through parameters. Changing ``rpd`` changes the valuation date
and nothing else has to be edited. That is what lets the Tier-2 spread calibration module price at
the *price observation date* instead of at ``VALUE_DATE``.

The I/O — reading curves, the ``FI_ZYC_VND_*.csv`` cache, ``CurveNode.get``, reading
``specs/hullwhite.json``, ``calc_rho``, the shock loop, writing Excel — **stays** in
``main_v2.py``. This module only receives curve DataFrames that are already prepared.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from callput import (
    CallPutTree, CurveLeg, PricingFlags, YieldCurve, compile_bond, hw_B,
)
from src.bond_schedule import CouponSchedule, adjust_following, build
from src.map_curve import MapCurve

__all__ = [
    "PricingConfig", "BondTermSheet", "ModelParams", "PricingResult",
    "refine_curve", "shift_x", "assert_curve_covers", "selected_fixing",
    "map_curves", "build_schedule", "build_legs", "build_tree", "price_bond",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PricingConfig:
    """
    Model constants, invariant across bonds and across valuation dates.

    Attributes:
        curve_folder (str): Folder holding the curve files.
        disc_convention (str): Day-count convention of the discount curve (LSCK).
        ref_convention (str): Day-count convention of the reference rate curve (LSTC).
        norm_step_days (float): Tree step in days for non-American bonds.
        ame_step_days (float): Tree step in days for American bonds; also the spacing of the
            American exercise ladder.
        min_step_days (float): Smallest tree step allowed, in days.
        fixing_lag_days (float): Days between the fixing date and the rate reset date.
        calendar_country (str): Key of the holiday calendar to use.
    """

    curve_folder: str
    disc_convention: str = "ACT/365"
    ref_convention: str = "ACT/365"
    norm_step_days: float = 21.0
    ame_step_days: float = 5.0
    min_step_days: float = 3.0
    # 0 means fixing date = rate reset date. bond_schedule.build tests `if fixing_lag_days`,
    # so 0.0 (falsy) goes straight to the no-subtraction branch.
    fixing_lag_days: float = 0.0
    calendar_country: str = "vnd"


@dataclass(frozen=True)
class ModelParams:
    """
    Calibrated Hull-White parameters for both factors.

    Attributes:
        a_r (float): Mean reversion of the discount-curve factor.
        sigma_r (float): Volatility of the discount-curve factor.
        a_L (float): Mean reversion of the reference-curve factor.
        sigma_L (float): Volatility of the reference-curve factor.
        rho (float): Correlation between the two factors.
    """

    a_r: float
    sigma_r: float
    a_L: float = 0.0
    sigma_L: float = 0.0
    rho: float = 0.0

    def with_vol_scale(self, k: float) -> "ModelParams":
        """
        Multiply both sigmas by ``k`` and return a new instance.

        Args:
            k (float): Scale factor applied to ``sigma_r`` and ``sigma_L``.

        Returns:
            ModelParams: A copy with scaled volatilities.
        """
        return replace(self, sigma_r=k * self.sigma_r, sigma_L=k * self.sigma_L)


# ---------------------------------------------------------------------------
# Term sheet
# ---------------------------------------------------------------------------
@dataclass
class BondTermSheet:
    """
    One term-sheet row together with its slice of the coupon schedule.

    The name is ``BondTermSheet`` rather than ``BondSpec`` because the earlier engine
    (``src/multi_hw_tree.py``, since removed) already had a class called ``BondSpec``.

    Every attribute here is **independent of the valuation date**. The date-dependent part lives
    only in :meth:`option_frames`.

    Attributes:
        frame (pd.DataFrame): One-row slice of ``bond_df``.
        coupon_schedule (pd.DataFrame): All coupon periods of the bond.
        holiday_calendar (dict): Holiday sets keyed by country.
        cfg (PricingConfig): Model constants.
        row (pd.Series): The single row of ``frame``.
        bond_id (str): Bond identifier.
        issue_date (pd.Timestamp): Issue date.
        style (str): Option style, lower-cased (e.g. ``"american"``).
        ref_curve_name (str | None): Reference curve name, lower-cased; ``None`` for a fixed bond.
        maturity_date (pd.Timestamp): Maturity, adjusted following on the holiday calendar.
        coupon_accrual (float): Coupon accrual value from the term sheet.
        face_value (float): Face value.
        group (str): Issuer group (TCPH).
        reset_dates (list): Rate reset dates; empty for a fixed bond.
    """

    frame: pd.DataFrame
    """One-row slice of ``bond_df``. It must be a DataFrame, not a Series:
    :meth:`CouponSchedule.build_reset_schedule` reads it as a frame, and ``row.to_frame().T``
    would coerce every column to ``object``, changing how pandas interprets dates."""

    coupon_schedule: pd.DataFrame
    """**All** coupon periods of the bond, filtered only by ``bond_id``. It must not be filtered
    by valuation date: ``build()`` steps back from the first period to construct
    ``accrual_start``, so it needs the past periods too."""

    holiday_calendar: dict
    cfg: PricingConfig

    row: pd.Series = field(init=False, repr=False)
    bond_id: str = field(init=False)
    issue_date: pd.Timestamp = field(init=False)
    style: str = field(init=False)
    ref_curve_name: "str | None" = field(init=False)
    maturity_date: pd.Timestamp = field(init=False)
    coupon_accrual: float = field(init=False)
    face_value: float = field(init=False)
    group: str = field(init=False)
    # fixed_rate: "list | None" = field(init=False, repr=False)
    reset_dates: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Derive the row-level attributes from ``frame``.

        Raises:
            ValueError: When ``frame`` does not have exactly one row.
        """
        if len(self.frame) != 1:
            raise ValueError(
                f"BondTermSheet cần đúng 1 dòng, nhận {len(self.frame)}"
            )
        row = self.frame.iloc[0]
        self.row = row
        self.bond_id = str(row["bond_id"])
        self.issue_date = pd.to_datetime(row["issue_date"])
        self.style = str(row["style"]).strip().lower()
        self.ref_curve_name = (
            None if pd.isna(row["ref_curve"]) else str(row["ref_curve"]).strip().lower()
        )
        # Same holiday calendar as CouponSchedule, so the principal date matches the last coupon
        # date. Passing the whole dict would make `date in holidays` compare against the keys
        # ('vnd') only, missing holidays and dodging just Saturdays/Sundays.
        self.maturity_date = adjust_following(
            pd.to_datetime(row["maturity_date"]),
            self.holiday_calendar.get(self.cfg.calendar_country, set()),
        )
        self.coupon_accrual = float(row["coupon_accrual"])
        self.face_value = float(row["face"])
        self.group = str(row["group"])

        if self.ref_curve_name is not None:
            # self.fixed_rate = None
            self.reset_dates = CouponSchedule(
                df=self.frame,
                holiday_calendar=self.holiday_calendar,
                country=self.cfg.calendar_country,
            ).build_reset_schedule()
        else:
            # self.fixed_rate = [
            #     float(x) for x in str(row["annual_coupon_rate"]).split(";")
            # ]
            self.reset_dates = []

    # -- construction -------------------------------------------------------
    @classmethod
    def from_bond_df(cls, bond_df, coupon_schedule_df, holiday_calendar, cfg,
                     *, iloc=None, bond_id=None) -> "BondTermSheet":
        """
        Build a term sheet from one row of ``bond_df``, selected by position or by id.

        Args:
            bond_df (pd.DataFrame): All term-sheet rows.
            coupon_schedule_df (pd.DataFrame): Coupon periods of all bonds.
            holiday_calendar (dict): Holiday sets keyed by country.
            cfg (PricingConfig): Model constants.
            iloc (int | None): Row position in ``bond_df``.
            bond_id (str | None): Bond identifier to select.

        Returns:
            BondTermSheet: The term sheet with its coupon periods filtered by ``bond_id``.

        Raises:
            ValueError: When not exactly one of ``iloc`` / ``bond_id`` is given.
        """
        if (iloc is None) == (bond_id is None):
            raise ValueError("chỉ định đúng một trong iloc / bond_id")
        frame = (bond_df.iloc[[iloc]] if iloc is not None
                 else bond_df[bond_df["bond_id"] == bond_id])
        key = frame.iloc[0]["bond_id"]
        return cls(
            frame=frame,
            coupon_schedule=coupon_schedule_df[coupon_schedule_df["bond_id"] == key],
            holiday_calendar=holiday_calendar,
            cfg=cfg,
        )

    # -- derived properties -------------------------------------------------
    @property
    def is_floating(self) -> bool:
        """
        Whether the bond references a floating rate curve.

        Returns:
            bool: True when ``ref_curve_name`` is set.
        """
        return self.ref_curve_name is not None

    @property
    def is_american(self) -> bool:
        """
        Whether the option style is American.

        Returns:
            bool: True when ``style == "american"``.
        """
        return self.style == "american"

    @property
    def step_days(self) -> float:
        """
        Tree step in days for this bond.

        Returns:
            float: ``cfg.ame_step_days`` for American bonds, otherwise ``cfg.norm_step_days``.
        """
        return self.cfg.ame_step_days if self.is_american else self.cfg.norm_step_days

    # -- option schedule ----------------------------------------------------
    def option_frames(self, rpd, *, anchor=None):
        """
        Call and put schedules at valuation date ``rpd``.

        The American exercise ladder is anchored at ``max(issue_date, anchor)``, and by default
        ``anchor = rpd``, which reproduces the old ``max(issue_date, VALUE_DATE)`` exactly. The
        consequence: moving ``rpd`` by one day shifts the **whole** ladder by one day, so the grid
        changes and the price moves by roughly the discretisation noise floor. The calibration
        module should pass ``anchor=spec.issue_date`` to keep the ladder fixed.

        Explicit exercise dates are read from ``call_exercise_dates`` / ``put_exercise_dates``
        (``;``-separated, ``%m/%d/%Y``). Strikes are expressed as a fraction of face value.

        Args:
            rpd (pd.Timestamp): Valuation date.
            anchor (pd.Timestamp | None): Start of the American ladder; defaults to ``rpd``.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: ``(call_df, put_df)`` with columns
            ``call_date``/``call_strike`` and ``put_date``/``put_strike``.
        """
        rpd = pd.Timestamp(rpd)
        anchor = rpd if anchor is None else pd.Timestamp(anchor)
        row, face = self.row, self.face_value
        freq = f"{str(self.cfg.ame_step_days)}D"

        def ladder(strike_col):
            """Evenly spaced American exercise dates with a constant strike."""
            dates = pd.date_range(
                start=max(self.issue_date, anchor),
                end=self.maturity_date,
                freq=freq,
            ).tolist()
            return dates, [float(row[strike_col]) / face] * len(dates)

        def explicit(date_col, strike_col):
            """Exercise dates and strikes listed explicitly in the term sheet."""
            dates = [pd.to_datetime(x.strip(), format="%m/%d/%Y")
                     for x in str(row[date_col]).split(";")]
            return dates, [float(x) / face for x in str(row[strike_col]).split(";")]

        if (self.is_american and pd.notna(row["call_strike"])
                and float(row["call_strike"]) > 0):
            call_dates, call_strikes = ladder("call_strike")
        elif pd.notna(row["call_exercise_dates"]) and str(row["call_exercise_dates"]).strip():
            call_dates, call_strikes = explicit("call_exercise_dates", "call_strike")
        else:
            call_dates, call_strikes = [], []

        if (self.is_american and pd.notna(row["put_strike"])
                and float(row["put_strike"]) > 0):
            put_dates, put_strikes = ladder("put_strike")
        elif pd.notna(row["put_exercise_dates"]) and str(row["put_exercise_dates"]).strip():
            put_dates, put_strikes = explicit("put_exercise_dates", "put_strike")
        else:
            put_dates, put_strikes = [], []

        call_df = pd.DataFrame({
            "call_date": pd.Series(call_dates, dtype="datetime64[ns]"),
            "call_strike": pd.Series(call_strikes, dtype="float64"),
        })
        put_df = pd.DataFrame({
            "put_date": pd.Series(put_dates, dtype="datetime64[ns]"),
            "put_strike": pd.Series(put_strikes, dtype="float64"),
        })
        return call_df, put_df


# ---------------------------------------------------------------------------
# Curves: grid refinement and shifting the x state
# ---------------------------------------------------------------------------
_FINE_MAX_YEARS = 30.0
_FINE_POINTS = 1500


def refine_curve(curve: YieldCurve, *, max_years=_FINE_MAX_YEARS,
                 n_points=_FINE_POINTS, tol=1e-9) -> YieldCurve:
    """
    The same curve on a denser grid, preserving values to within 1 ULP.

    ``YieldCurve`` interpolates linearly in zero-rate space and extrapolates flat. Linearly
    interpolating a piecewise-linear function on a grid that *contains* every original node gives
    back exactly the same function, mathematically.

    Measured: at the original pillars the result matches exactly; between pillars it differs by
    exactly **1 ULP** (1.4e-17 absolute, 2.2e-16 relative) because ``np.interp`` rounds differently
    when it passes through an intermediate grid point. This is the smallest possible error of the
    refinement, not something that can be loosened.

    Args:
        curve (YieldCurve): Curve to refine.
        max_years (float): Last point of the geometric fine grid, in years.
        n_points (int): Number of points in the fine grid.
        tol (float): Fine points closer than this to an original node are dropped.

    Returns:
        YieldCurve: Curve on the union of the original nodes and the fine grid.
    """
    mats = np.asarray(curve.maturities, float)
    fine = np.geomspace(1.0 / 365.0, max_years, n_points)
    fine = fine[np.min(np.abs(fine[:, None] - mats[None, :]), axis=1) > tol]
    grid = np.unique(np.concatenate([mats, fine]))
    return YieldCurve.from_zero_rates(grid, curve.zero_rate(grid))


def shift_x(curve: YieldCurve, a: float, delta: float, *, refine=True) -> YieldCurve:
    """
    Shift the Hull-White factor ``x`` by ``delta``, expressed on the curve::

        P(0,T)  ->  P(0,T) · exp(−B_a(T)·delta)
        R(T)    ->  R(T) + delta · B_a(T)/T

    The spread is **not flat**: the transmission factor ``B_a(T)/T`` equals 1 at the short end and
    tends to ``1/(aT)`` at the long end.

    The fine grid is **mandatory**, not a nicety: ``B_a(T)/T`` is curved while ``YieldCurve``
    interpolates linearly, so applying the spread only at the 18 pillars and leaving the rest to
    ``np.interp`` would be off by up to a few bp between the sparse long-end pillars. The spread
    calibration would then converge to the interpolation error rather than to the market price.

    ``delta == 0`` returns the **same** object untouched, so every existing path stays
    bit-identical.

    Args:
        curve (YieldCurve): Curve to shift.
        a (float): Hull-White mean reversion of the factor.
        delta (float): Shift of ``x``.
        refine (bool): Refine the grid with :func:`refine_curve` before shifting.

    Returns:
        YieldCurve: The shifted curve, or ``curve`` itself when ``delta`` is zero.
    """
    if not delta:
        return curve
    c = refine_curve(curve) if refine else curve
    return c.shifted(delta * hw_B(a, c.maturities) / c.maturities)


# ---------------------------------------------------------------------------
# Mapping curves at a date, with a guard against silent fallback
# ---------------------------------------------------------------------------
def assert_curve_covers(rpd, df, label: str) -> None:
    """
    Fail if ``rpd`` is earlier than every row of ``df``.

    ``MapCurve.map_curve`` falls back to ``df.index.min()`` **silently** when no row is
    ``<= rpd``. At ``VALUE_DATE`` that never happens; at a past price observation date it does, and
    it turns "no data" into a number that looks perfectly plausible. The guard lives here instead
    of changing ``MapCurve``.

    Args:
        rpd (pd.Timestamp): Date the curve is mapped at.
        df (pd.DataFrame): Curve history indexed by date.
        label (str): Curve name used in the error message.

    Raises:
        ValueError: When ``rpd`` is before the first row of ``df``.
    """
    start = pd.Timestamp(pd.DatetimeIndex(df.index).min())
    if pd.Timestamp(rpd) < start:
        raise ValueError(
            f"{label}: rpd={pd.Timestamp(rpd).date()} sớm hơn dòng đầu tiên "
            f"{start.date()} — MapCurve sẽ âm thầm dùng {start.date()}."
        )


def selected_fixing(spec: BondTermSheet, rpd):
    """
    The fixing date ``build()`` will pick for the first future coupon period.

    Replicates the logic in ``bond_schedule.build`` so it can be guarded up front; if that function
    changes, this copy must change with it. Returns ``None`` when it does not apply — including
    when the nearest future period is still in the fixed phase of a switching bond (read from
    ``coupon_type`` in ``spec.coupon_schedule``, no longer using ``spec.is_floating`` as a proxy).

    Args:
        spec (BondTermSheet): The bond.
        rpd (pd.Timestamp): Valuation date.

    Returns:
        pd.Timestamp | None: The fixing date, net of ``cfg.fixing_lag_days``, or ``None``.
    """
    if not spec.is_floating:
        return None
    rpd = pd.Timestamp(rpd)
    sched = spec.coupon_schedule
    future = sched[sched["pay_date"] > rpd].sort_values("pay_date")
    if future.empty:
        return None
    first = future.iloc[0]
    if first["coupon_type"] != "float":
        return None
    cands = [r for r in spec.reset_dates if r <= first["start_date"]]
    if not cands:
        return None
    sel = max(cands)
    lag = spec.cfg.fixing_lag_days
    return sel - pd.Timedelta(days=lag) if lag else sel


def map_curves(spec, rpd, disc_df, ref_df, *, require_curve_history=False):
    """
    Curve DataFrames -> ``(disc_curve, ref_curve)`` at ``rpd``.

    Args:
        spec (BondTermSheet): The bond; supplies conventions and the fixing date.
        rpd (pd.Timestamp): Valuation date.
        disc_df (pd.DataFrame): Discount curve (LSCK) history.
        ref_df (pd.DataFrame | None): Reference rate curve (LSTC) history; ``None`` for a fixed
            bond.
        require_curve_history (bool): Guard with :func:`assert_curve_covers` that both curves
            (and the reference curve at the fixing date) have data on or before the date.

    Returns:
        tuple: ``(disc_curve, ref_curve)``; ``ref_curve`` is ``None`` when ``ref_df`` is.

    Raises:
        ValueError: When ``require_curve_history`` is set and a curve has no history that early.
    """
    if require_curve_history:
        assert_curve_covers(rpd, disc_df, "đường chiết khấu")
        if ref_df is not None:
            assert_curve_covers(rpd, ref_df, "đường tham chiếu")
            fix = selected_fixing(spec, rpd)
            if fix is not None:
                # get_known_rate re-maps the reference curve at the fixing date, which is earlier
                # than rpd; this is where the fallback is most likely to trigger.
                assert_curve_covers(fix, ref_df, "đường tham chiếu @fixing")
    cfg = spec.cfg
    disc_curve = MapCurve(
        rpd=rpd, df=disc_df, convention=cfg.disc_convention
    ).map_curve()
    ref_curve = (
        MapCurve(rpd=rpd, df=ref_df, convention=cfg.ref_convention).map_curve()
        if ref_df is not None else None
    )
    return disc_curve, ref_curve


# ---------------------------------------------------------------------------
# Build the schedule, the legs and the tree
# ---------------------------------------------------------------------------
def build_schedule(spec, rpd, *, apply_floor=True,
                   apply_cap=True, call_df=None, put_df=None,
                   option_anchor=None, ref_df_raw=None):
    """
    Build the cash-flow schedule of ``spec`` at ``rpd``.

    Replaces the old ``build_sched`` closure; every captured variable is now a parameter.

    Args:
        spec (BondTermSheet): The bond.
        rpd (pd.Timestamp): Valuation date.
        apply_floor (bool): Apply the coupon floor.
        apply_cap (bool): Apply the coupon cap.
        call_df (pd.DataFrame | None): Call schedule; from :meth:`BondTermSheet.option_frames`
            when omitted.
        put_df (pd.DataFrame | None): Put schedule; from :meth:`BondTermSheet.option_frames`
            when omitted.
        option_anchor (pd.Timestamp | None): Anchor of the American exercise ladder.
        ref_df_raw (pd.DataFrame | None): Raw reference curve data, passed through to ``build``.

    Returns:
        The schedule returned by ``bond_schedule.build``.
    """
    if call_df is None or put_df is None:
        c, p = spec.option_frames(rpd, anchor=option_anchor)
        call_df = c if call_df is None else call_df
        put_df = p if put_df is None else put_df
    return build(
        rpd=rpd,
        face=spec.face_value,
        maturity_date=spec.maturity_date,
        coupon_schedule_df=spec.coupon_schedule,
        ref_dates=spec.reset_dates,
        call_df=call_df,
        put_df=put_df,
        # fixed_rate=spec.fixed_rate,
        fixing_lag_days=spec.cfg.fixing_lag_days,
        ref_curve_name=spec.ref_curve_name,
        curve_folder=str(spec.cfg.curve_folder),
        ref_convention=spec.cfg.ref_convention,
        apply_floor=apply_floor,
        apply_cap=apply_cap,
        ref_df_raw=ref_df_raw,
    )


def build_legs(disc_curve, ref_curve, params, cfg, *, disc_delta=0.0,
               ref_delta=0.0, refine=True):
    """
    Build the discount and reference curve legs, with an optional shift in ``x`` space.

    Replaces the old ``legs`` function. ``*_delta`` shifts the factor ``x`` (OAS / Tier-2 capital
    spread), producing a curved spread ``δ·B_a(τ)/τ`` on the zero rates — not a flat additive
    spread. The shift lives here because only here is it certain that ``a_r`` goes with the
    discount curve and ``a_L`` with the reference curve.

    Args:
        disc_curve (YieldCurve): Discount curve (LSCK).
        ref_curve (YieldCurve | None): Reference rate curve (LSTC); ``None`` for a fixed bond.
        params (ModelParams): Hull-White parameters.
        cfg (PricingConfig): Model constants.
        disc_delta (float): Shift of the discount factor ``x``.
        ref_delta (float): Shift of the reference factor ``x``.
        refine (bool): Refine the grid before shifting (see :func:`shift_x`).

    Returns:
        tuple[CurveLeg, CurveLeg | None]: ``(disc_leg, ref_leg)``; ``ref_leg`` is ``None`` when
        ``ref_curve`` is.
    """
    disc = shift_x(disc_curve, params.a_r, disc_delta, refine=refine)
    disc_leg = CurveLeg(disc, params.a_r, params.sigma_r, cfg.disc_convention)
    if ref_curve is None:
        return disc_leg, None
    ref = shift_x(ref_curve, params.a_L, ref_delta, refine=refine)
    return disc_leg, CurveLeg(ref, params.a_L, params.sigma_L, cfg.ref_convention)


def build_tree(spec, rpd, *, disc_df, params, ref_df=None,
               apply_floor=True, apply_cap=True, step_days=None, min_step=None,
               option_anchor=None, disc_delta=0.0,
               ref_delta=0.0, refine_on_delta=True,
               require_curve_history=False, sched=None, ref_df_raw=None):
    """
    One term-sheet row + one date ``rpd`` -> ``(CompiledBond, CallPutTree)``.

    The order of operations matches the old version — map curves, build schedule, compile, build
    legs — so the order of printed warning lines is unchanged too.

    Args:
        spec (BondTermSheet): The bond.
        rpd (pd.Timestamp): Valuation date.
        disc_df (pd.DataFrame): Discount curve (LSCK) history.
        params (ModelParams): Hull-White parameters.
        ref_df (pd.DataFrame | None): Reference rate curve (LSTC) history.
        apply_floor (bool): Apply the coupon floor.
        apply_cap (bool): Apply the coupon cap.
        step_days (float | None): Tree step; defaults to ``spec.step_days``.
        min_step (float | None): Minimum tree step; defaults to ``cfg.min_step_days``.
        option_anchor (pd.Timestamp | None): Anchor of the American exercise ladder.
        disc_delta (float): Shift of the discount factor ``x``.
        ref_delta (float): Shift of the reference factor ``x``.
        refine_on_delta (bool): Refine the curve grid when shifting.
        require_curve_history (bool): Guard against the silent ``MapCurve`` fallback.
        sched (optional): Pre-built schedule; built with :func:`build_schedule` when omitted.
        ref_df_raw (pd.DataFrame | None): Raw reference curve data, passed through to ``build``.

    Returns:
        tuple[CompiledBond, CallPutTree]: The compiled bond and its single- or multi-curve tree.
    """
    cfg = spec.cfg
    disc_curve, ref_curve = map_curves(
        spec, rpd, disc_df, ref_df, require_curve_history=require_curve_history
    )
    if sched is None:
        sched = build_schedule(
            spec, rpd, apply_floor=apply_floor,
            apply_cap=apply_cap, option_anchor=option_anchor,
            ref_df_raw=ref_df_raw,
        )
    bond = compile_bond(
        sched,
        step_days=spec.step_days if step_days is None else step_days,
        min_step=cfg.min_step_days if min_step is None else min_step,
    )
    disc_leg, ref_leg = build_legs(
        disc_curve, ref_curve, params, cfg,
        disc_delta=disc_delta, ref_delta=ref_delta, refine=refine_on_delta,
    )
    if ref_leg is None:
        return bond, CallPutTree.single_curve(bond, disc_leg)
    return bond, CallPutTree.multi_curve(bond, disc_leg, ref_leg, rho=params.rho)


@dataclass(frozen=True)
class PricingResult:
    """
    Result of pricing one bond at one valuation date.

    Attributes:
        bond_id (str): Bond identifier.
        rpd (pd.Timestamp): Valuation date.
        full_price (float): Price with the embedded options.
        straight_price (float): Price without the options.
        diff (float): ``full_price - straight_price``.
        bond (object): The compiled bond, for inspection.
        tree (object): The ``CallPutTree`` that produced ``full_price``.
    """

    bond_id: str
    rpd: pd.Timestamp
    full_price: float
    straight_price: float
    diff: float
    bond: object = field(repr=False, default=None)
    tree: object = field(repr=False, default=None)


def price_bond(spec, rpd, **kw) -> PricingResult:
    """
    Full pricing: the price with options, the straight price, and the difference.

    Keeps the two separate calls ``price()`` and ``decompose()`` as in the old version.
    ``decompose()`` recomputes ``full`` itself, so numerically they could be merged, but merging is
    a change that needs proving while keeping them does not.

    Args:
        spec (BondTermSheet): The bond.
        rpd (pd.Timestamp): Valuation date.
        **kw: Passed through to :func:`build_tree`.

    Returns:
        PricingResult: Full and straight prices from the same tree.
    """
    bond, tree = build_tree(spec, rpd, **kw)
    full = tree.price()
    straight = tree.decompose()["straight"]
    return PricingResult(
        spec.bond_id, pd.Timestamp(rpd), full, straight, full - straight, bond, tree
    )


def price_bond_layered(spec, rpd, *, delta_full, delta_straight,
                       **kw) -> PricingResult:
    """
    The price with options and the straight price on **two different discount curves**.

    ``price_bond`` takes both numbers from one tree, so both sit on the same curve. Here the two
    legs are deliberately split:

    ====================  ==================  ======================
    Type                  ``delta_full``      ``delta_straight``
    ====================  ==================  ======================
    Not Tier-2 capital    ``OAS``             ``0``
    Tier-2 capital        ``OAS + delta``     ``delta``
    ====================  ==================  ======================

    The `LB_G*` curve is built from **option-free** bonds, so it is exactly the right curve for
    the straight leg; the leg with options must add ``OAS`` to match the market price of that
    optioned bond itself. The bottom row is exactly "the Tier-2 capital tree with options
    **minus** ``OAS``".

    **Consequence to keep in mind when reading results**: ``diff = full - straight`` now contains
    both the option value **and** the difference due to ``OAS``; it is no longer the pure option
    value. The ``decompose()`` identity only holds on a single curve, so do not combine ``diff``
    here with the components of ``decompose()``.

    The two trees share the lattice geometry: :class:`FactorLattice` reads only ``a``, ``sigma``
    and the date grid, **not the curve**. So the discretisation error still cancels when taking
    the difference, just as when both legs are on one tree.

    Args:
        spec (BondTermSheet): The bond.
        rpd (pd.Timestamp): Valuation date.
        delta_full (float): Discount-factor shift for the leg with options.
        delta_straight (float): Discount-factor shift for the straight leg.
        **kw: Passed through to :func:`build_tree` (must not contain ``disc_delta``).

    Returns:
        PricingResult: ``full_price`` and ``tree`` from the ``delta_full`` tree; ``straight_price``
        from the ``delta_straight`` tree.

    Raises:
        TypeError: When ``disc_delta`` is passed in ``kw``.
        AssertionError: When the two trees do not share the same date grid.
    """
    if "disc_delta" in kw:
        raise TypeError(
            "price_bond_layered nhận delta_full / delta_straight, không nhận "
            "disc_delta — truyền nhầm sẽ cho hai chân cùng một đường mà không báo"
        )
    d_full, d_straight = float(delta_full), float(delta_straight)

    bond, tree = build_tree(spec, rpd, disc_delta=d_full, **kw)
    full = tree.price()

    if d_full == d_straight:
        # One curve -> one tree. This branch yields EXACTLY the two numbers of `price_bond`,
        # because `decompose()["straight"]` is `price(PricingFlags.none())`. It is the regression
        # gate: with no OAS, the result must match the old version bit for bit.
        straight = tree.price(PricingFlags.none())
    else:
        bond_s, tree_s = build_tree(spec, rpd, disc_delta=d_straight, **kw)
        if not np.array_equal(bond.days, bond_s.days):
            raise AssertionError(
                "hai cây không cùng lưới ngày — hiệu full - straight sẽ lẫn sai "
                "số rời rạc hoá thay vì chỉ còn giá trị quyền chọn"
            )
        straight = tree_s.price(PricingFlags.none())

    return PricingResult(
        spec.bond_id, pd.Timestamp(rpd), full, straight, full - straight, bond, tree
    )
