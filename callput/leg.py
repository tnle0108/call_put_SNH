"""Curve legs and the Hull-White affine primitives.

Lifted from :mod:`src.hw_tree` -- ``_B``, ``_integrated_variance`` and
``_ou_variance`` become module-level functions so both the one- and two-factor
engines share them, and the ``a -> 0`` limits are carried over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .curve import YieldCurve

# Only conventions LINEAR in actual days may serve as a diffusion clock.
#
# 30/360 counts 01/02->01/03 and 01/03->01/04 both as 30 days although they span
# 28 and 31 actual days, so the model would accrue equal variance over unequal
# real time.  It is a convention for counting *money* (accrual, accrued interest),
# not for measuring *time*.  ACT/365 and ACT/360 differ only by a constant factor,
# which is why running each leg on its own clock is exactly a reparametrisation
# of (a, sigma) -- see :meth:`CurveLeg.retimed`.
ACT_FAMILY: dict[str, float] = {
    "ACT/365": 365.0,
    "ACT/365F": 365.0,
    "ACT/360": 360.0,
    "ACT/365.25": 365.25,
}


def hw_B(a: float, tau):
    """Hull-White B(t, t+tau) = (1 - e^{-a tau}) / a, a->0 -> tau."""
    tau = np.asarray(tau, dtype=float)
    if a < 1e-8:
        return tau
    return -np.expm1(-a * tau) / a


def hw_G(a: float, sigma: float, tau):
    """Cumulative log-bond variance sigma^2 int_0^tau B(u)^2 du.

    Closed form sigma^2/a^2 [tau + (2/a)e^{-a tau} - (1/2a)e^{-2a tau} - 3/2a];
    as a->0 this tends to sigma^2 tau^3 / 3.
    """
    tau = np.asarray(tau, dtype=float)
    if a < 1e-6:
        return sigma**2 * tau**3 / 3.0
    return (sigma**2 / a**2) * (
        tau
        + (2.0 / a) * np.exp(-a * tau)
        - (1.0 / (2.0 * a)) * np.exp(-2.0 * a * tau)
        - 3.0 / (2.0 * a)
    )


def ou_variance(a: float, sigma: float, dt: float) -> float:
    """Conditional variance sigma^2/(2a) (1 - e^{-2a dt}), a->0 -> sigma^2 dt."""
    if a < 1e-8:
        return sigma**2 * dt
    return sigma**2 * (-np.expm1(-2.0 * a * dt)) / (2.0 * a)


@dataclass
class CurveLeg:
    """A curve together with the Hull-White factor driving it.

    Each leg owns its day-count convention and the two legs of a tree need not
    agree.  Correlation between legs carries no time unit and is shared unchanged.

    Every public method takes **integer day offsets** from the valuation date,
    never year fractions: the day grid is the single source of truth and each leg
    converts it privately, so a caller cannot leak the wrong convention in.

    ``a`` and ``sigma`` must have been calibrated **in this leg's own clock**;
    :meth:`retimed` performs the conversion when they were not.
    """

    curve: YieldCurve
    a: float
    sigma: float
    dcc: str = "ACT/365"

    denom: float = field(init=False)

    def __post_init__(self) -> None:
        if self.dcc not in ACT_FAMILY:
            raise ValueError(
                f"{self.dcc!r} is not linear in actual days and cannot be a "
                f"diffusion clock (it is a money-counting convention). Use one of "
                f"{sorted(ACT_FAMILY)} for the model clock; the quoted curve may "
                f"still have been bootstrapped under {self.dcc!r}."
            )
        self.denom = ACT_FAMILY[self.dcc]
        if self.a < 0:
            raise ValueError("mean reversion a must be >= 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")

    # -- time ---------------------------------------------------------------
    def t(self, days):
        """Year fraction of ``days`` (integer offsets) in this leg's clock."""
        return np.asarray(days, dtype=float) / self.denom

    def discount(self, days):
        """Market ``P(0, days)``."""
        return self.curve.discount(self.t(days))

    def retimed(self, curve: YieldCurve | None = None, dcc: str | None = None) -> "CurveLeg":
        """The same factor expressed on another clock, and optionally another curve.

        With ``k = d_old / d_new`` the clock stretches as ``t -> k t``.  The short
        rate is itself quoted per unit of time, so it contracts as ``r -> r / k``
        to leave every discount factor unchanged, and the factor follows::

            a -> a / k          sigma -> sigma / k**1.5

        The exponent is 1.5, not 0.5: half of it is the usual square-root-of-time
        rescaling, the other half is the short rate's own change of units.  The
        check is that ``G``, and hence the convexity adjustment, comes out exactly
        invariant -- with ``sigma / sqrt(k)`` it is wrong by a factor ``k**2``,
        about 2.8% between ACT/365 and ACT/360.  ``B`` scales as ``k B``, which
        cancels against ``z -> z / k`` so that ``B z`` is invariant too.
        """
        new_dcc = self.dcc if dcc is None else dcc
        if new_dcc not in ACT_FAMILY:
            raise ValueError(f"{new_dcc!r} cannot be a diffusion clock")
        k = self.denom / ACT_FAMILY[new_dcc]
        return CurveLeg(
            curve=self.curve if curve is None else curve,
            a=self.a / k,
            sigma=self.sigma / k**1.5,
            dcc=new_dcc,
        )

    # -- the single affine primitive ---------------------------------------
    def affine_zcb(self, from_days: int, to_days: int) -> tuple[float, float]:
        """Coefficients ``(A, B)`` of ``P(t, T | z) = A * exp(-B * z)``.

        ``A`` splits into a curve part and a model part::

            A = [P(T) / P(t)] * exp((G(tau) - G(T) + G(t)) / 2)

        The curve part is a ratio of discount factors -- a pure number carrying no
        convention, because the year fraction used to annualise the forward rate
        cancels against the one used to discount it back.  Only ``G`` and ``B``
        consume this leg's clock, and they are exactly the terms answering "how
        much uncertainty accumulates over this interval".

        This is the one primitive behind the rollback discount factor, the coupon
        index and the in-advance present value of a deferred coupon; only the
        curve, the tenor and the downstream compounding convention differ.
        """
        t = float(self.t(from_days))
        T = float(self.t(to_days))
        tau = T - t
        if tau <= 0:
            raise ValueError("to_days must be strictly after from_days")
        ratio = self.curve.discount(T) / self.curve.discount(t)
        conv = (
            hw_G(self.a, self.sigma, tau)
            - hw_G(self.a, self.sigma, T)
            + hw_G(self.a, self.sigma, t)
        ) / 2.0
        return float(ratio * np.exp(conv)), float(hw_B(self.a, tau))

    # -- geometry helpers ---------------------------------------------------
    def variance(self, step_days: int) -> float:
        """One-step conditional variance of the factor."""
        return ou_variance(self.a, self.sigma, step_days / self.denom)

    def terminal_std(self, horizon_days: int) -> float:
        """Std of the factor at the horizon; drives the state-space truncation."""
        T = float(self.t(horizon_days))
        if self.a < 1e-8:
            return self.sigma * np.sqrt(T)
        return self.sigma * np.sqrt(-np.expm1(-2.0 * self.a * T) / (2.0 * self.a))
