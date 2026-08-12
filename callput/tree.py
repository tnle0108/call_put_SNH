"""Backward induction for a bond with embedded call, put, cap and floor.

Follows ``MultiCurveHWTree.price`` / ``.decompose`` but adds the one thing a
recombining tree cannot express on its own: a coupon whose rate was set at an
earlier node.  A bond resetting every 12 months and paying every 6 months fixes
one rate that governs several payments, and by the time the rollback reaches a
payment date the lattice has forgotten which node the fixing came from -- a node
at the payment date is reachable from several fixing levels.

The fix is to carry the fixing as a **label**: between the last payment it
governs and the fixing date itself, the value array grows a leading axis holding
one copy per possible fixed level.  The axis does not branch and has no
probability of its own; the copies differ only in the cash injected into them and
in the exercise decisions they consequently make.  At the fixing date the label
becomes fact -- it is determined by that node's own level -- so the axis collapses
along its diagonal and the rollback returns to its normal width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import Lattice
from .leg import CurveLeg
from .schedule import CompiledBond, FixingGroup, Period


@dataclass
class PricingFlags:
    """Toggle embedded features on/off for twin-run decomposition."""

    floor: bool = True
    cap: bool = True
    call: bool = True
    put: bool = True

    @classmethod
    def none(cls) -> "PricingFlags":
        return cls(floor=False, cap=False, call=False, put=False)


class CallPutTree:
    """Prices a :class:`~callput.schedule.CompiledBond` on a Hull-White lattice.

    Build it through :meth:`single_curve` (fixed and step-up coupons, one factor)
    or :meth:`multi_curve` (any floating coupon, two factors).
    """

    def __init__(
        self,
        bond: CompiledBond,
        disc_leg: CurveLeg,
        ref_leg: CurveLeg | None = None,
        rho: float = 0.0,
        n_sigma: float = 6.0,
        n_fix: int | None = None,
    ) -> None:
        self.bond = bond
        self.disc = disc_leg
        self.ref = disc_leg if ref_leg is None else ref_leg
        if n_fix is not None and n_fix < 5:
            # Two or three labels cannot resolve the kinks the exercise tests put
            # into the value's dependence on the fixed rate, and the result is
            # quietly wrong rather than merely coarse.  21 labels measured around
            # a basis point of face; below that, use the exact grid.
            raise ValueError(
                f"n_fix={n_fix} is too coarse to interpolate the fixing axis; "
                f"pass at least 5, or None for the exact grid"
            )
        self.n_fix = n_fix

        legs = [disc_leg] if ref_leg is None else [disc_leg, ref_leg]
        self.lat = Lattice(legs, bond.days, rho=rho, n_sigma=n_sigma)
        # The fixing factor is always the last state axis: the reference factor
        # when there are two, the discount factor when there is one.  That is what
        # lets the collapse be one call to np.diagonal for both engines.
        self.fix_factor = self.lat.factors[-1]

        days = bond.days
        self.step_discount = []
        for i in range(bond.n_steps):
            a_coef, b_coef = disc_leg.affine_zcb(int(days[i]), int(days[i + 1]))
            self.step_discount.append(
                a_coef * np.exp(-b_coef * self.lat.factors[0].x[i])
            )
        self._fit_to_curve()

    def _fit_to_curve(self) -> None:
        """Rescale each step's discount so the lattice reprices the curve exactly.

        The affine node rates of :class:`src.hw_tree.HullWhiteTree` reproduce the
        market curve only up to the tree's discretisation of the Gaussian -- a
        residual of a few tenths of a basis point of face, which
        ``curve_repricing_error`` was there to report.  It is small but systematic,
        and it does not cancel between a base and a shocked run.

        Hull's remedy, at a scalar per step: carry Arrow-Debreu prices forward and
        scale step ``i``'s discount by ``P(0, t_{i+1}) / sum_j Q_j D_j``.  A single
        multiplier per step suffices because the correction is a parallel shift of
        that step's rate, and it leaves the branch probabilities untouched.

        Only the discount factor is involved, so the induction runs on the first
        factor alone; the correlation matrices have zero row and column sums, so a
        two-factor joint reproduces that factor's marginal exactly.
        """
        factor = self.lat.factors[0]
        weights = np.ones(1)
        for i in range(self.bond.n_steps):
            target = self.disc.discount(int(self.bond.days[i + 1]))
            self.step_discount[i] = self.step_discount[i] * (
                target / float((weights * self.step_discount[i]).sum())
            )
            flow = weights * self.step_discount[i]
            nxt = np.zeros(int(factor.n[i + 1]))
            for branch in range(3):
                np.add.at(
                    nxt, factor.child[i][:, branch], flow * factor.prob[i][:, branch]
                )
            weights = nxt

    # -- constructors -------------------------------------------------------
    @classmethod
    def single_curve(
        cls, bond: CompiledBond, disc_leg: CurveLeg, n_sigma: float = 6.0
    ) -> "CallPutTree":
        """One factor, one curve: fixed-rate bonds, step-ups included.

        Every period must carry a ``fixed_rate``; a step-up is simply a different
        rate per period.  With no index there is no fixing to remember and no cap
        or floor to exercise, so the fixing axis never opens.
        """
        floating = [p for p in bond.periods if p.is_float]
        if floating:
            raise ValueError(
                f"{len(floating)} period(s) float (first pays on day "
                f"{floating[0].src.pay_day}); a floating coupon needs the "
                f"reference factor -- use CallPutTree.multi_curve"
            )
        return cls(bond, disc_leg, ref_leg=None, n_sigma=n_sigma)

    @classmethod
    def multi_curve(
        cls,
        bond: CompiledBond,
        disc_leg: CurveLeg,
        ref_leg: CurveLeg,
        rho: float,
        n_sigma: float = 6.0,
        n_fix: int | None = None,
    ) -> "CallPutTree":
        """Two factors: discounting and the coupon index, correlated by ``rho``.

        Fixed and floating periods may be mixed -- a bond fixed for two years then
        floating runs on the plain two-dimensional array until the first reset.
        """
        return cls(
            bond, disc_leg, ref_leg=ref_leg, rho=rho, n_sigma=n_sigma, n_fix=n_fix
        )

    # -- pricing ------------------------------------------------------------
    def price(self, flags: PricingFlags | None = None) -> float:
        """Present value today (dirty) of the bond."""
        flags = flags or PricingFlags()
        bond = self.bond
        n = bond.maturity_step

        value = np.full(self.lat.shape(n), bond.face, dtype=float)
        # group: FixingGroup | None = None
        # f_idx = f_vals = None
        active: list[tuple[FixingGroup, np.ndarray, np.ndarray]] = []

        for i in range(n, -1, -1):
            if i < n:
                value = self.lat.rollback(value, i, self.step_discount[i])

            for k, (grp, f_idx, f_vals) in enumerate(active):
                if grp.fix_step == i:
                    value = self._collapse_at(value, grp, f_idx, k)
                    del active[k]
                    break

            # if group is not None and i == group.fix_step:
            #     value = self._collapse(value, group, f_idx)
            #     group = f_idx = f_vals = None

            opening = bond.expand_at.get(i)
            if opening is not None:
                # if group is not None:  # pragma: no cover - compile_bond forbids it
                #     raise AssertionError("nested fixing groups")
                # group = opening
                f_idx, f_vals = self._fixing_grid(opening)
                value = np.repeat(value[np.newaxis], f_vals.size, axis=0)
                active.insert(0, (opening, f_idx, f_vals))

            # value = self._exercise(value, i, group, f_vals, flags)
            # cash = self._coupons(i, group, f_vals, flags)
            value = self._exercise(value, i, active, flags)
            cash = self._coupons(i, active, flags)
            if cash is not None:
                value = value + cash

        return float(value[(0,) * self.lat.n_factors])

    def decompose(self) -> dict[str, float]:
        """Isolate each embedded option by toggling flags.

        ``interaction`` is the residual that linear attribution cannot reach.  For
        a bond that is both floored and callable it is not small: the floor makes
        the bond dearer and therefore likelier to be called, so the two features
        eat into each other.  It is reported rather than spread silently over the
        components.
        """
        straight = self.price(PricingFlags.none())
        full = self.price(PricingFlags())

        def only(**kw: bool) -> float:
            base = dict(floor=False, cap=False, call=False, put=False)
            return self.price(PricingFlags(**{**base, **kw}))

        v_floor = only(floor=True) - straight
        v_cap = straight - only(cap=True)
        v_call = straight - only(call=True)
        v_put = only(put=True) - straight
        return {
            "straight": straight,
            "floor": v_floor,
            "cap": v_cap,
            "call": v_call,
            "put": v_put,
            "interaction": full - (straight + v_floor - v_cap - v_call + v_put),
            "full": full,
        }

    # -- fixing axis --------------------------------------------------------
    def _fixing_grid(self, group: FixingGroup) -> tuple[np.ndarray, np.ndarray]:
        """Levels of the fixing factor to carry as labels through this group."""
        size = int(self.fix_factor.n[group.fix_step])
        if self.n_fix is None or self.n_fix >= size:
            idx = np.arange(size)
        else:
            idx = np.unique(
                np.rint(np.linspace(0, size - 1, self.n_fix)).astype(int)
            )
        return idx, self.fix_factor.x[group.fix_step][idx]

    def _collapse(
        self, value: np.ndarray, group: FixingGroup, f_idx: np.ndarray
    ) -> np.ndarray:
        """Drop the fixing axis: at the fixing date the label is the node's level.

        ``out[..., m] = value[f(m), ..., m]`` -- only the diagonal survives.  The
        off-diagonal entries were never wasted: at the payment dates in between, a
        node is reachable from several fixing levels and every one of those
        answers had to be available.
        """
        size = int(self.fix_factor.n[group.fix_step])
        if value.shape[-1] != size:  # pragma: no cover - defensive
            raise AssertionError(
                f"fixing factor has {size} levels at step {group.fix_step} but the "
                f"value array's last axis is {value.shape[-1]}; np.diagonal would "
                f"truncate this silently"
            )
        if f_idx.size == size:
            return np.ascontiguousarray(np.diagonal(value, axis1=0, axis2=-1))

        # Coarse label grid: interpolate back onto every level.  Between two
        # exercise dates the value is linear in the fixed rate and each min/max
        # adds a single kink, so linear interpolation is well inside the tree's
        # own discretisation error.
        pos = np.interp(
            np.arange(size, dtype=float),
            f_idx.astype(float),
            np.arange(f_idx.size, dtype=float),
        )
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, f_idx.size - 1)
        weight = pos - lo
        low = np.diagonal(value[lo], axis1=0, axis2=-1)
        high = np.diagonal(value[hi], axis1=0, axis2=-1)
        return (1.0 - weight) * low + weight * high

    def _collapse_at(self, value:np.ndarray, group: FixingGroup, f_idx:np.ndarray, axis: int) -> np.ndarray:
        size = int(self.fix_factor.n[group.fix_step])
        if value.shape[-1] != size:
            raise AssertionError(
                f"fixing factor has {size} levels at step {group.fix_step} but the "
                f"value array's last axis is {value.shape[-1]}; collapse would "
                f"truncate this silently"
            )

        if f_idx.size == size:
            return np.ascontiguousarray(np.diagonal(value, axis1=axis, axis2=-1))

        pos = np.interp(
            np.arange(size, dtype=float),
            f_idx.astype(float),
            np.arange(f_idx.size, dtype=float),
        )

        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, f_idx.size -1)
        weight = pos - lo
        v_lo = np.take(value, lo, axis = axis)
        v_hi = np.take(value, hi, axis = axis)
        low = np.diagonal(v_lo, axis1=axis, axis2=-1)
        high = np.diagonal(v_hi, axis1=axis, axis2=-1)
        return (1 - weight) * low + weight * high
    # -- cash and exercise --------------------------------------------------
    def _index_rate(
        self, fix_step: int, period: Period, z_vals: np.ndarray, flags: PricingFlags
    ) -> np.ndarray:
        """Coupon rate set by the fixing at ``fix_step``, over the given levels.

        Called **once per fixing**, never once per payment, which is what keeps a
        single contractual floor a single option: a 12-month reset paying twice a
        year observes the floor once and both coupons inherit the result.

        The cap and floor bound the **coupon** rate, so the margin goes in first::

            rate = min(max(L + margin, floor), cap)
        """
        src = period.src
        if src.fixed_rate is not None:
            rate = np.full(np.shape(z_vals), src.fixed_rate, dtype=float)
        else:
            start = int(self.bond.days[fix_step])
            a_coef, b_coef = self.ref.affine_zcb(start, start + src.ref_tenor_days)
            bond_price = a_coef * np.exp(-b_coef * z_vals)
            rate = (1.0 / bond_price - 1.0) / src.ref_delta + src.margin
        if flags.floor and src.floor is not None:
            rate = np.maximum(rate, src.floor)
        if flags.cap and src.cap is not None:
            rate = np.minimum(rate, src.cap)
        return rate

    def _rate_of(
        self,
        period: Period,
        i: int,
        # group: FixingGroup | None,
        # f_vals: np.ndarray | None,
        active,
        flags: PricingFlags,
    ):
        """Rate of ``period`` as seen at step ``i``, shaped to broadcast onto V."""
        found = self._find_active(active, period.fix_step) if period.is_deferred else None
        if found is not None:
            k, f_vals = found
            rate = self._index_rate(period.fix_step, period, f_vals, flags)
            ndim = len(active) + self.lat.n_factors
            shape = [1] * ndim
            shape[k] = f_vals.size
            return rate.reshape(shape)

        if period.fix_step is None:
            return float(period.src.fixed_rate)

        # deferred = (
        #     group is not None
        #     and period.is_deferred
        #     and period.fix_step == group.fix_step
        # )
        # if deferred:
        #     rate = self._index_rate(period.fix_step, period, f_vals, flags)
        #     return rate.reshape((-1,) + (1,) * self.lat.n_factors)

        if period.fix_step > i:
            raise ValueError(
                f"the period paying on day {period.src.pay_day} fixes on day "
                f"{period.src.fixing_day}, after the exercise date on day "
                f"{int(self.bond.days[i])}; its accrued interest is not known "
                f"there. An in-arrears period exercised mid-period needs a "
                f"contractual rule this engine does not assume."
            )
        if period.fix_step < i:  # pragma: no cover - defensive
            # Reading the fixing factor's levels at an earlier step while the
            # value array is indexed at this one is precisely the amnesia the
            # label axis exists to avoid, and it would fail silently.
            raise AssertionError(
                f"the period paying on day {period.src.pay_day} was fixed at step "
                f"{period.fix_step} but its fixing axis is not open at step {i}"
            )
        # Fixed at this very node: read straight off the fixing factor's levels.
        return self._index_rate(
            period.fix_step, period, self.fix_factor.x[period.fix_step], flags
        )

    def _accrued(
        self,
        i: int,
        # group: FixingGroup | None,
        # f_vals: np.ndarray | None,
        active,
        flags: PricingFlags,
    ):
        """Accrued interest at step ``i``; zero on a payment date.

        Uses the capped/floored rate, not the raw index: getting this wrong shifts
        the exercise boundary exactly in the scenarios where the floor is biting.
        """
        period = self.bond.accrue_at.get(i)
        if period is None:
            return 0.0
        src = period.src
        elapsed = int(self.bond.days[i]) - src.accrual_start_day
        span = src.pay_day - src.accrual_start_day
        fraction = src.accrual * elapsed / span
        return self._rate_of(period, i, active, flags) * fraction * self.bond.face

    def _exercise(
        self,
        value: np.ndarray,
        i: int,
        # group: FixingGroup | None,
        # f_vals: np.ndarray | None,
        active,
        flags: PricingFlags,
    ) -> np.ndarray:
        call = self.bond.call.get(i) if flags.call else None
        put = self.bond.put.get(i) if flags.put else None
        if call is None and put is None:
            return value
        # Strikes are clean, quoted as a fraction of face.
        accrued = self._accrued(i, active, flags)
        if call is not None:
            value = np.minimum(call * self.bond.face + accrued, value)
        if put is not None:
            value = np.maximum(put * self.bond.face + accrued, value)
        return value

    def _coupons(
        self,
        i: int,
        # group: FixingGroup | None,
        # f_vals: np.ndarray | None,
        active,
        flags: PricingFlags,
    ):
        """Cash paid at step ``i``, or ``None``.

        Added *after* the exercise test: a coupon falling due on the exercise date
        is paid whether or not the bond is called.  Coupons fixed here but paid
        later are the opposite case -- they ride inside the continuation value and
        die with it, which is why the axis opens before the test, not after.
        """
        due = self.bond.pay_at.get(i)
        if not due:
            return None
        total = None
        for period in due:
            rate = self._rate_of(period, i, active, flags)
            cash = rate * period.src.accrual * self.bond.face
            total = cash if total is None else total + cash
        return total

    # -- diagnostics --------------------------------------------------------
    def zero_price(self, step: int | None = None) -> float:
        """Lattice price of a unit zero maturing at ``step`` -- a pure rollback."""
        step = self.bond.maturity_step if step is None else step
        value = np.ones(self.lat.shape(step))
        for i in range(step - 1, -1, -1):
            value = self.lat.rollback(value, i, self.step_discount[i])
        return float(value[(0,) * self.lat.n_factors])

    def diagnostics(self) -> dict[str, object]:
        """Health checks that do not need a reference implementation."""
        maturity = int(self.bond.days[self.bond.maturity_step])
        out: dict[str, object] = {
            "n_steps": self.bond.n_steps,
            "max_nodes": self.lat.max_nodes,
            "degenerate_nodes": self.lat.n_degenerate,
            "levels_per_factor": [int(f.n.max()) for f in self.lat.factors],
            "min_step_days": int(self.bond.step_days().min()),
            "max_step_days": int(self.bond.step_days().max()),
            "curve_repricing_error": abs(
                self.zero_price() - self.disc.discount(maturity)
            ),
        }
        if self.lat.n_factors == 2:
            mid = self.bond.n_steps // 2
            out["realized_correlation"] = self.lat.realized_correlation(mid)
            out["target_correlation"] = self.lat.rho
        return out

    def _find_active(self, active, fix_step):
        for k, (grp, f_idx, f_vals) in enumerate(active):
            if grp.fix_step == fix_step:
                return k, f_vals
        return None
