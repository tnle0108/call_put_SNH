from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from quantmr.leg import Leg, FixedLeg, FloatingLeg
from quantmr.config import load_data, load_spec
from quantmr.curve.zerocurve import SimZeroCurve
from quantmr.fx.fx import SimFX
from quantmr.latentstate import LatentState
from quantmr.model.shortrate.hullwhite import HullWhite


class RateDerivative(ABC):

    def __init__(self, pay_leg: Leg, rec_leg: Leg):
        self.pay_leg = pay_leg
        self.rec_leg = rec_leg
        self._par_rate: np.float64 | None = None
        self._par_spread: np.float64 | None = None

    @property
    def start_date(self) -> np.datetime64:
        return min(
            self.pay_leg.schedule.astarts[0],
            self.rec_leg.schedule.astarts[0],
        )

    @property
    def as_of(self) -> np.datetime64:
        as_of = (
            self.pay_leg._disc_curve_as_of
            or self.rec_leg._disc_curve_as_of
            or self.start_date
        )
        return np.datetime64(as_of).astype("datetime64[D]")

    @property
    def par_rate(self) -> np.float64 | None:
        return self._par_rate

    @property
    def par_spread(self) -> np.float64 | None:
        return self._par_spread

    @abstractmethod
    def pv(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        ...

    def cashflow(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return {
            "pay": self.pay_leg.cashflow(value_date, **kwargs),
            "rec": self.rec_leg.cashflow(value_date, **kwargs),
        }

    def _resolve_leg(self, leg: str, leg_type: type) -> Leg:
        if leg == "auto":
            if isinstance(self.pay_leg, leg_type):
                return self.pay_leg
            if isinstance(self.rec_leg, leg_type):
                return self.rec_leg
            raise ValueError(f"No {leg_type.__name__} found")
        if leg == "pay":
            return self.pay_leg
        if leg == "rec":
            return self.rec_leg
        raise ValueError(f"Unknown leg selector: {leg!r}")

    def solve_rate(
        self,
        leg: str = "auto",
        value_date: str | np.datetime64 | None = None,
    ) -> np.float64:
        target_leg = self._resolve_leg(leg, FixedLeg)
        assert isinstance(target_leg, FixedLeg)
        vd = value_date or self.start_date
        target_leg._fixed_rate = 0.0
        pv0 = float(self.pv(vd).ravel()[0])
        target_leg._fixed_rate = 1.0
        pv1 = float(self.pv(vd).ravel()[0])
        dpv = pv1 - pv0
        if np.isclose(dpv, 0.0):
            raise ValueError("PV is insensitive to fixed rate")
        rate = -pv0 / dpv
        target_leg._fixed_rate = rate
        self._par_rate = np.float64(rate)
        return self._par_rate

    def solve_spread(
        self,
        leg: str = "auto",
        value_date: str | np.datetime64 | None = None,
    ) -> np.float64:
        target_leg = self._resolve_leg(leg, FloatingLeg)
        assert isinstance(target_leg, FloatingLeg)
        vd = value_date or self.start_date
        target_leg._spread = 0.0
        pv0 = float(self.pv(vd).ravel()[0])
        target_leg._spread = 1.0
        pv1 = float(self.pv(vd).ravel()[0])
        dpv = pv1 - pv0
        if np.isclose(dpv, 0.0):
            raise ValueError("PV is insensitive to spread")
        spread = -pv0 / dpv
        target_leg._spread = spread
        self._par_spread = np.float64(spread)
        return self._par_spread

    def _gather_factors(self) -> dict[str, float]:
        factors: dict[str, float] = {}
        for leg_ in (self.pay_leg, self.rec_leg):
            if leg_._disc_curve:
                name = leg_._disc_curve.lower()
                if name not in factors:
                    hw = HullWhite.get(name)
                    factors[name] = float(hw.a)
            if isinstance(leg_, FloatingLeg) and leg_._ref_curve:
                name = leg_._ref_curve.lower()
                if name not in factors:
                    hw = HullWhite.get(name)
                    factors[name] = float(hw.a)
        return factors

    @staticmethod
    def _load_rho(factor_names: list[str]) -> np.ndarray:
        n = len(factor_names)
        corr_data = load_data("correlation").get("corr")
        if corr_data is None or not isinstance(corr_data, pd.DataFrame):
            return np.eye(n, dtype=np.float64)
        if "Corr" in corr_data.columns:
            corr_data = corr_data.set_index("Corr")
        available = [
            f
            for f in factor_names
            if f in corr_data.index and f in corr_data.columns
        ]
        if len(available) == n:
            return corr_data.loc[factor_names, factor_names].values.astype(
                np.float64
            )
        rho = np.eye(n, dtype=np.float64)
        for i, fi in enumerate(factor_names):
            for j, fj in enumerate(factor_names):
                if fi in corr_data.index and fj in corr_data.columns:
                    rho[i, j] = np.float64(corr_data.loc[fi, fj])
        return rho

    @staticmethod
    def _get_leg_epsilon(
        leg_: Leg,
        eps_dict: dict,
        t_value_idx: int,
        t_fix_idx: int | None,
        factor_idx: dict[str, int],
    ) -> dict:
        kw: dict = {}
        if leg_._disc_curve:
            name = leg_._disc_curve.lower()
            if name in factor_idx:
                eps = eps_dict.get((t_value_idx, factor_idx[name]))
                if eps is not None:
                    kw["disc_epsilon"] = eps
        if isinstance(leg_, FloatingLeg) and leg_._ref_curve:
            name = leg_._ref_curve.lower()
            if name in factor_idx:
                eps = eps_dict.get((t_value_idx, factor_idx[name]))
                if eps is not None:
                    kw["ref_epsilon"] = eps
                if (
                    t_fix_idx is not None
                    and t_fix_idx != t_value_idx
                ):
                    fix_eps = eps_dict.get((t_fix_idx, factor_idx[name]))
                    if fix_eps is not None:
                        kw["fix_epsilon"] = fix_eps
        return kw

    def _combine_mtm(
        self,
        rec_pv: np.ndarray,
        pay_pv: np.ndarray,
        d: np.datetime64 | None = None,
        eps_dict: dict | None = None,
        eps_t: int | None = None,
        factor_idx: dict[str, int] | None = None,
    ) -> np.ndarray:
        return rec_pv - pay_pv

    def _find_fix_date(
        self,
        d: np.datetime64,
    ) -> np.datetime64 | None:
        for leg_ in (self.pay_leg, self.rec_leg):
            if not isinstance(leg_, FloatingLeg):
                continue
            astarts = leg_.schedule.astarts
            aends = leg_.schedule.aends
            mask = (astarts <= d) & (d <= aends)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                j = int(idx[0])
                if astarts[j] == d and j > 0:
                    fd = astarts[j - 1]
                else:
                    fd = astarts[j]
                return np.datetime64(fd).astype("datetime64[D]")
        return None

    def mtm(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        n_samples: int = 10_000,
        state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        vd = np.atleast_1d(
            np.asarray(value_date, dtype="datetime64[D]")
        ).ravel()
        factors = self._gather_factors()
        factor_names = list(factors.keys())
        factor_a = np.array(
            [factors[n] for n in factor_names], dtype=np.float64
        )
        factor_idx = {name: i for i, name in enumerate(factor_names)}
        dc = self.pay_leg.daycount
        as_of = self.as_of
        rho = (
            self._load_rho(factor_names) if factor_names else np.empty((0, 0))
        )
        if isinstance(state, np.random.Generator):
            rng = state
        elif isinstance(state, int):
            rng = np.random.default_rng(state)
        else:
            rng = np.random.default_rng()
        results: list[np.ndarray] = []
        for d in vd:
            t_value = float(
                np.asarray(dc.yearfrac(as_of, d), dtype=np.float64).ravel()[0]
            )
            if t_value <= 0 or len(factor_names) == 0:
                pay_pv = self.pay_leg.pv(d)
                rec_pv = self.rec_leg.pv(d)
                mtm_val = self._combine_mtm(rec_pv, pay_pv, d=d)
                mtm_row = mtm_val.squeeze(axis=0)
                if np.ndim(mtm_row) == 0:
                    mtm_row = np.full(
                        n_samples, float(mtm_row), dtype=np.float64
                    )
                results.append(np.atleast_1d(mtm_row))
                continue
            t_fix_idx: int | None = None
            fix_date = self._find_fix_date(d)
            if fix_date is not None:
                t_fix = float(
                    np.asarray(
                        dc.yearfrac(as_of, fix_date), dtype=np.float64
                    ).ravel()[0]
                )
                if t_fix > 0 and not np.isclose(t_fix, t_value):
                    times = np.array([t_fix, t_value], dtype=np.float64)
                    t_fix_idx = 0
                    t_value_idx = 1
                else:
                    times = np.array([t_value], dtype=np.float64)
                    t_value_idx = 0
            else:
                times = np.array([t_value], dtype=np.float64)
                t_value_idx = 0
            ls = LatentState(t=times, a=factor_a, rho=rho)
            eps_dict = ls.generate_epsilon_dict(
                n_samples=n_samples, state=rng
            )
            pay_kw = self._get_leg_epsilon(
                self.pay_leg, eps_dict, t_value_idx, t_fix_idx, factor_idx
            )
            rec_kw = self._get_leg_epsilon(
                self.rec_leg, eps_dict, t_value_idx, t_fix_idx, factor_idx
            )
            pay_pv = self.pay_leg.pv(d, **pay_kw)
            rec_pv = self.rec_leg.pv(d, **rec_kw)
            mtm_val = self._combine_mtm(
                rec_pv, pay_pv,
                d=d, eps_dict=eps_dict, eps_t=t_value_idx, factor_idx=factor_idx,
            )
            mtm_row = mtm_val.squeeze(axis=0)
            if np.ndim(mtm_row) == 0:
                mtm_row = np.full(
                    n_samples, float(mtm_row), dtype=np.float64
                )
            results.append(np.atleast_1d(mtm_row))
            for key in list(SimZeroCurve.SIMZEROCURVE_CACHE):
                szc = SimZeroCurve.SIMZEROCURVE_CACHE.pop(key)
                szc.__dict__.pop("_knots", None)
            for key in list(SimFX.SIMFX_CACHE):
                del SimFX.SIMFX_CACHE[key]
        return np.stack(results, axis=0)


    @staticmethod
    def _build_leg(
        side: str,
        spec: dict,
        *,
        start: str | np.datetime64 | None = None,
        end: str | np.datetime64 | None = None,
        term: str | None = None,
        freq: str | None = None,
        disc_curve_as_of: str | np.datetime64 | None = None,
        ref_curve_as_of: str | np.datetime64 | None = None,
        disc_is_sim: bool = False,
        ref_is_sim: bool = False,
        notional: float | np.float64 | None = None,
        fixed_rate: float | np.float64 | None = None,
        spread: float | np.float64 | None = None,
        notional_exchange: bool | None = None,
    ) -> FixedLeg | FloatingLeg:
        is_fixed = spec.get(f"{side}_is_fixed", False)
        ne = (
            notional_exchange
            if notional_exchange is not None
            else spec.get("notional_exchange", False)
        )
        common = dict(
            start=start,
            end=end,
            term=term,
            freq=freq,
            calendar=spec.get("calendar"),
            busadjust=spec.get("busadjust"),
            eom=spec.get("eom", False),
            notional=notional,
            currency=spec.get(f"{side}_currency"),
            convention=spec.get(f"{side}_convention"),
            compounding=spec.get(f"{side}_compounding"),
            notional_exchange=ne,
            disc_curve=spec.get(f"{side}_disc_curve"),
            disc_is_sim=disc_is_sim,
            disc_curve_as_of=disc_curve_as_of,
        )
        if is_fixed:
            rate = (
                fixed_rate
                if fixed_rate is not None
                else spec.get(f"{side}_fixed_rate")
            )
            return FixedLeg(fixed_rate=rate, **common)
        else:
            return FloatingLeg(
                ref_curve=spec.get(f"{side}_ref_curve"),
                ref_is_sim=ref_is_sim,
                ref_curve_as_of=ref_curve_as_of,
                spread=spread,
                **common,
            )

    @classmethod
    def get(
        cls,
        name: str,
        *,
        start: str | np.datetime64 | None = None,
        end: str | np.datetime64 | None = None,
        term: str | None = None,
        freq: str | None = None,
        disc_curve_as_of: str | np.datetime64 | None = None,
        ref_curve_as_of: str | np.datetime64 | None = None,
        disc_is_sim: bool = False,
        ref_is_sim: bool = False,
        pay_notional: float | np.float64 | None = None,
        rec_notional: float | np.float64 | None = None,
        pay_fixed_rate: float | np.float64 | None = None,
        rec_fixed_rate: float | np.float64 | None = None,
        pay_spread: float | np.float64 | None = None,
        rec_spread: float | np.float64 | None = None,
        notional_exchange: bool | None = None,
        fx_as_of: str | np.datetime64 | None = None,
        solve: bool = True,
    ) -> "RateDerivative":
        spec = load_spec("ratederivative").get(name.lower(), {})
        if not spec:
            raise ValueError(f"Unknown derivative name: {name!r}")
        deri_type = spec.get("type", "irs")

        pay_leg = cls._build_leg(
            "pay",
            spec,
            start=start,
            end=end,
            term=term,
            freq=freq,
            disc_curve_as_of=disc_curve_as_of,
            ref_curve_as_of=ref_curve_as_of,
            disc_is_sim=disc_is_sim,
            ref_is_sim=ref_is_sim,
            notional=pay_notional,
            fixed_rate=pay_fixed_rate,
            spread=pay_spread,
            notional_exchange=notional_exchange,
        )
        rec_leg = cls._build_leg(
            "rec",
            spec,
            start=start,
            end=end,
            term=term,
            freq=freq,
            disc_curve_as_of=disc_curve_as_of,
            ref_curve_as_of=ref_curve_as_of,
            disc_is_sim=disc_is_sim,
            ref_is_sim=ref_is_sim,
            notional=rec_notional,
            fixed_rate=rec_fixed_rate,
            spread=rec_spread,
            notional_exchange=notional_exchange,
        )

        from quantmr.ratederivative.irs import IRS
        from quantmr.ratederivative.ccs import (
            CCS,
            FixedFloatCCS,
            FixedFixedCCS,
            FloatFloatCCS,
        )

        if deri_type == "irs":
            obj = IRS(pay_leg, rec_leg)
        elif deri_type == "ccs":
            ccs_type = spec.get("ccs_type", "floatfloat")
            fx_pair = spec.get("fx_pair")
            ccs_cls_map = {
                "fixedfloat": FixedFloatCCS,
                "floatfixed": FixedFloatCCS,
                "floatfloat": FloatFloatCCS,
                "fixedfixed": FixedFixedCCS,
            }
            ccs_cls = ccs_cls_map.get(ccs_type, CCS)
            obj = ccs_cls(pay_leg, rec_leg, fx_pair=fx_pair, fx_as_of=fx_as_of)
        else:
            raise ValueError(f"Unknown derivative type: {deri_type!r}")

        if solve:
            if isinstance(pay_leg, FixedLeg) and pay_leg._fixed_rate is None:
                obj.solve_rate(leg="pay")
            elif isinstance(rec_leg, FixedLeg) and rec_leg._fixed_rate is None:
                obj.solve_rate(leg="rec")
            if isinstance(pay_leg, FloatingLeg) and pay_leg._spread is None:
                obj.solve_spread(leg="pay")
            elif isinstance(rec_leg, FloatingLeg) and rec_leg._spread is None:
                obj.solve_spread(leg="rec")

        return obj
