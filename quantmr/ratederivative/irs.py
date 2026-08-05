import numpy as np
from quantmr.ratederivative.ratederivative import RateDerivative
from quantmr.leg import FixedLeg, FloatingLeg


def irs_pv(
    rec_leg,
    pay_leg,
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
    return rec_leg.pv(value_date, **rec_kw) - pay_leg.pv(value_date, **pay_kw)


class IRS(RateDerivative):

    def __init__(
        self,
        pay_leg: FixedLeg | FloatingLeg,
        rec_leg: FixedLeg | FloatingLeg,
    ):
        if pay_leg.currency != rec_leg.currency:
            raise ValueError(
                f"IRS legs must share the same currency, "
                f"got pay={pay_leg.currency} rec={rec_leg.currency}"
            )
        super().__init__(pay_leg, rec_leg)

    def pv(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        disc_epsilon: np.ndarray | None = None,
        ref_epsilon: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        return irs_pv(
            self.rec_leg, self.pay_leg,
            value_date, disc_epsilon, ref_epsilon,
            **kwargs,
        )
