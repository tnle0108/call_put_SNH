import numpy as np
from quantmr.leg.leg import Leg
from quantmr.curve import ZeroCurve, SimZeroCurve


def _calculate_cashflow(
    rate, spread, notional, period_yearfrac, compounding="simple"
):
    match compounding:
        case "continuous":
            cf = np.asarray(
                notional * (np.exp(rate * np.asarray(period_yearfrac)) - 1.0),
                dtype=np.float64,
            )
        case "simple":
            cf = np.asarray(
                notional * rate * period_yearfrac,
                dtype=np.float64,
            )
        case "annually":
            cf = np.asarray(
                notional * ((1.0 + rate) ** np.asarray(period_yearfrac) - 1.0),
                dtype=np.float64,
            )
        case _:
            raise ValueError(f"Unsupported compounding method: {compounding}")
    cf = cf + np.asarray(notional * spread * period_yearfrac, dtype=np.float64)
    return cf


class FixedLeg(Leg):

    def __init__(
        self,
        fixed_rate: float | np.float64 | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._fixed_rate = fixed_rate

    @property
    def fixed_rate(self) -> np.float64:
        return np.float64(self._fixed_rate if self._fixed_rate is not None else np.nan)

    def cashflow(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        cf = _calculate_cashflow(
            rate=self.fixed_rate,
            spread=0.0,
            notional=self.notional,
            period_yearfrac=self.period_yearfrac,
            compounding=self.compound.compounding,
        )
        cf = self._add_final_notional(cf)
        past = self._past_mask(value_date)
        return np.where(past, np.nan, cf)


class FloatingLeg(Leg):

    def __init__(
        self,
        ref_curve: str | None = None,
        ref_is_sim: bool = False,
        ref_curve_as_of: str | np.datetime64 | None = None,
        spread: float | np.float64 | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ref_curve = ref_curve
        self._ref_is_sim = ref_is_sim
        self._ref_curve_as_of = ref_curve_as_of
        self._spread = spread

    @property
    def spread(self) -> np.float64:
        return np.float64(self._spread if self._spread is not None else 0.0)

    def _ref_curve_at(
        self, date: np.datetime64, epsilon: np.ndarray | None = None
    ) -> ZeroCurve | SimZeroCurve:
        if self._ref_curve is None:
            raise ValueError("ref_curve must be set")
        if self._ref_is_sim:
            if self._ref_curve_as_of is None:
                raise ValueError(
                    "ref_curve_as_of must be set for simulation curves"
                )
            return SimZeroCurve.get(
                self._ref_curve, self._ref_curve_as_of, date,
                epsilon=epsilon,
            )
        return ZeroCurve.get(self._ref_curve, self._ref_curve_as_of or date)

    def cashflow(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        ref_epsilon: np.ndarray | None = None,
        fix_epsilon: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        vd = np.atleast_1d(
            np.asarray(value_date, dtype="datetime64[D]")
        ).ravel()
        astarts = self.schedule.astarts
        aends = self.schedule.aends
        fix_eps = fix_epsilon if fix_epsilon is not None else ref_epsilon
        slices: list[np.ndarray] = []
        for d in vd:
            proj = self._ref_curve_at(d, epsilon=ref_epsilon)
            rates = proj.forward_rate(astarts, aends)
            fix_idx = np.where((astarts <= d) & (d <= aends))[0]
            for j_item in fix_idx:
                j = int(j_item)
                if astarts[j] == d and j > 0:
                    fix_date = np.datetime64(astarts[j - 1].item()).astype(
                        "datetime64[D]"
                    )
                else:
                    fix_date = np.datetime64(astarts[j].item()).astype(
                        "datetime64[D]"
                    )
                if self._ref_is_sim:
                    fix = self._ref_curve_at(fix_date, epsilon=fix_eps)
                else:
                    fix = ZeroCurve.get(self._ref_curve, fix_date)
                fix_rate = fix.forward_rate(astarts[j], aends[j])
                if rates.ndim == 1:
                    rates[j] = np.asarray(fix_rate).ravel()[0]
                else:
                    rates[j, :] = np.asarray(fix_rate).ravel()
            slices.append(rates)
        all_rates = np.stack(slices, axis=0)
        yf = self.period_yearfrac
        if all_rates.ndim > 2:
            yf = yf[..., None]
        cf = _calculate_cashflow(
            rate=all_rates,
            spread=self.spread,
            notional=self.notional,
            period_yearfrac=yf,
            compounding=self.compound.compounding,
        )
        cf = self._add_final_notional(cf)
        past = self._past_mask(value_date)
        if cf.ndim > past.ndim:
            past = past[..., None]
        return np.where(past, np.nan, cf)
