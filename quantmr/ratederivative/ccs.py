import numpy as np
from quantmr.ratederivative.ratederivative import RateDerivative
from quantmr.leg import FixedLeg, FloatingLeg
from quantmr.fx import FX, SimFX


class CCS(RateDerivative):

    def __init__(
        self,
        pay_leg: FixedLeg | FloatingLeg,
        rec_leg: FixedLeg | FloatingLeg,
        fx_pair: str | None = None,
        fx_as_of: str | np.datetime64 | None = None,
    ):
        super().__init__(pay_leg, rec_leg)
        self._fx_pair = fx_pair
        self._fx_as_of = fx_as_of

    @property
    def _pay_is_base(self) -> bool:
        pair = (self._fx_pair or "").lower()
        return pair.startswith(self.pay_leg.currency.lower())

    @property
    def _rec_is_base(self) -> bool:
        pair = (self._fx_pair or "").lower()
        return pair.startswith(self.rec_leg.currency.lower())

    def _gather_factors(self) -> dict[str, float]:
        factors = super()._gather_factors()
        if self._fx_pair:
            fx_factor = f"spot_{self._fx_pair.lower()}"
            factors[fx_factor] = 0.0
        return factors

    def _fx_at(self, value_date: np.datetime64) -> FX:
        if self._fx_pair is None:
            raise ValueError("fx_pair must be set for CCS")
        as_of = self._fx_as_of or value_date
        return FX.get((self._fx_pair, as_of))

    def _fx_spot(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
    ) -> np.ndarray:
        vd = np.atleast_1d(
            np.asarray(value_date, dtype="datetime64[D]")
        ).ravel()
        return np.array([self._fx_at(d).spot for d in vd], dtype=np.float64)

    def pv(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        disc_epsilon: np.ndarray | None = None,
        ref_epsilon: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        rec_kw = {**kwargs}
        pay_kw = {**kwargs}
        if disc_epsilon is not None:
            rec_kw["disc_epsilon"] = disc_epsilon
            pay_kw["disc_epsilon"] = disc_epsilon
        if ref_epsilon is not None:
            rec_kw["ref_epsilon"] = ref_epsilon
            pay_kw["ref_epsilon"] = ref_epsilon
        rec_pv = self.rec_leg.pv(value_date, **rec_kw)
        pay_pv = self.pay_leg.pv(value_date, **pay_kw)
        fx_raw = self._fx_spot(value_date)
        rec_fx = fx_raw if self._rec_is_base else np.ones_like(fx_raw)
        pay_fx = fx_raw if self._pay_is_base else np.ones_like(fx_raw)
        ndim = max(rec_pv.ndim, pay_pv.ndim)
        if ndim > rec_fx.ndim:
            rec_fx = rec_fx[..., None]
        if ndim > pay_fx.ndim:
            pay_fx = pay_fx[..., None]
        return rec_pv * rec_fx - pay_pv * pay_fx

    def _combine_mtm(
        self,
        rec_pv: np.ndarray,
        pay_pv: np.ndarray,
        d: np.datetime64,
        eps_dict: dict,
        eps_t: int | None,
        factor_idx: dict[str, int],
    ) -> np.ndarray:
        if self._fx_pair is None:
            return rec_pv - pay_pv
        fx_factor = f"spot_{self._fx_pair.lower()}"
        fx_eps = None
        if fx_factor in factor_idx and eps_t is not None:
            fx_eps = eps_dict.get((eps_t, factor_idx[fx_factor]))
        fx_as_of = self._fx_as_of or self.as_of
        sfx = SimFX.get((self._fx_pair, fx_as_of, d), epsilon=fx_eps)
        fx_rate = sfx.spot
        rec_fx = fx_rate if self._rec_is_base else np.float64(1.0)
        pay_fx = fx_rate if self._pay_is_base else np.float64(1.0)
        ndim = max(rec_pv.ndim, pay_pv.ndim)
        if ndim > np.ndim(rec_fx):
            rec_fx = np.atleast_1d(rec_fx)[None, :]
        if ndim > np.ndim(pay_fx):
            pay_fx = np.atleast_1d(pay_fx)[None, :]
        return rec_pv * rec_fx - pay_pv * pay_fx


def FixedFloatCCS(
    pay_leg: FixedLeg | FloatingLeg,
    rec_leg: FixedLeg | FloatingLeg,
    fx_pair: str | None = None,
    fx_as_of: str | np.datetime64 | None = None,
) -> CCS:
    if pay_leg.currency == rec_leg.currency:
        raise ValueError(
            f"CCS legs must have different currencies, "
            f"both are {pay_leg.currency}"
        )
    fixed_count = sum(isinstance(leg, FixedLeg) for leg in (pay_leg, rec_leg))
    floating_count = sum(
        isinstance(leg, FloatingLeg) for leg in (pay_leg, rec_leg)
    )
    if fixed_count != 1 or floating_count != 1:
        raise ValueError(
            "FixedFloatCCS requires exactly one FixedLeg and one FloatingLeg"
        )
    return CCS(pay_leg, rec_leg, fx_pair=fx_pair, fx_as_of=fx_as_of)


def FixedFixedCCS(
    pay_leg: FixedLeg,
    rec_leg: FixedLeg,
    fx_pair: str | None = None,
    fx_as_of: str | np.datetime64 | None = None,
) -> CCS:
    if pay_leg.currency == rec_leg.currency:
        raise ValueError(
            f"CCS legs must have different currencies, "
            f"both are {pay_leg.currency}"
        )
    if not isinstance(pay_leg, FixedLeg) or not isinstance(rec_leg, FixedLeg):
        raise ValueError("FixedFixedCCS requires both legs to be FixedLeg")
    return CCS(pay_leg, rec_leg, fx_pair=fx_pair, fx_as_of=fx_as_of)


def FloatFloatCCS(
    pay_leg: FloatingLeg,
    rec_leg: FloatingLeg,
    fx_pair: str | None = None,
    fx_as_of: str | np.datetime64 | None = None,
) -> CCS:
    if pay_leg.currency == rec_leg.currency:
        raise ValueError(
            f"CCS legs must have different currencies, "
            f"both are {pay_leg.currency}"
        )
    if not isinstance(pay_leg, FloatingLeg) or not isinstance(
        rec_leg, FloatingLeg
    ):
        raise ValueError(
            "FloatFloatCCS requires both legs to be FloatingLeg"
        )
    return CCS(pay_leg, rec_leg, fx_pair=fx_pair, fx_as_of=fx_as_of)
