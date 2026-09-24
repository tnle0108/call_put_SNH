"""Zero curves.

Standalone: this package does not import from ``src/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class YieldCurve:
    """A continuously-compounded zero curve with flat-rate extrapolation.

    Parameters
    ----------
    maturities:
        Strictly increasing tenors in **years**, all > 0.  The unit is whatever
        day-count convention the curve was bootstrapped with; the
        :class:`~callput.leg.CurveLeg` that owns this curve is responsible for
        converting day offsets into that same unit.
    zero_rates:
        Continuously-compounded zero rates (decimals) for each maturity.

    Interpolation is linear in zero-rate space; outside the quoted range the
    nearest endpoint rate is held flat.  ``P(0, t) = exp(-R(0, t) * t)``.
    """

    maturities: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.zero_rates = np.asarray(self.zero_rates, dtype=float)
        if self.maturities.ndim != 1 or self.zero_rates.ndim != 1:
            raise ValueError("maturities and zero_rates must be 1-D arrays")
        if self.maturities.shape != self.zero_rates.shape:
            raise ValueError("maturities and zero_rates must have the same length")
        if self.maturities.size == 0:
            raise ValueError("curve needs at least one point")
        if np.any(self.maturities <= 0):
            raise ValueError("maturities must be strictly positive")
        if np.any(np.diff(self.maturities) <= 0):
            raise ValueError("maturities must be strictly increasing")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_zero_rates(cls, maturities, zero_rates) -> "YieldCurve":
        return cls(np.asarray(maturities, float), np.asarray(zero_rates, float))

    @classmethod
    def from_discount_factors(cls, maturities, discount_factors) -> "YieldCurve":
        mats = np.asarray(maturities, float)
        dfs = np.asarray(discount_factors, float)
        if np.any(dfs <= 0):
            raise ValueError("discount factors must be positive")
        return cls(mats, -np.log(dfs) / mats)

    @classmethod
    def flat(cls, rate: float, horizon: float = 50.0) -> "YieldCurve":
        """A flat curve -- handy for tests and for sanity-checking a bond."""
        return cls(np.array([1e-6, horizon]), np.array([rate, rate]))

    @classmethod
    def from_dataframe(
        cls,
        df,
        maturity_col: str = "maturity",
        rate_col: str | None = "zero_rate",
        discount_col: str | None = None,
    ) -> "YieldCurve":
        frame = df.sort_values(maturity_col)
        mats = frame[maturity_col].to_numpy(dtype=float)
        if discount_col is not None:
            return cls.from_discount_factors(mats, frame[discount_col].to_numpy(float))
        if rate_col is None:
            raise ValueError("supply either rate_col or discount_col")
        return cls.from_zero_rates(mats, frame[rate_col].to_numpy(float))

    # -- queries -----------------------------------------------------------
    def zero_rate(self, t):
        t_arr = np.asarray(t, dtype=float)
        rate = np.interp(t_arr, self.maturities, self.zero_rates)
        return rate if t_arr.ndim else float(rate)

    def discount(self, t):
        t_arr = np.asarray(t, dtype=float)
        df = np.exp(-self.zero_rate(t_arr) * t_arr)
        return df if t_arr.ndim else float(df)

    def forward_rate(self, t1, t2):
        return np.log(self.discount(t1) / self.discount(t2)) / (t2 - t1)

    def shifted(self, bump) -> "YieldCurve":
        """Parallel or per-pillar bump, for key-rate and Basel shock scenarios.

        The tree grid never depends on rates, so a shifted curve reprices on the
        exact same lattice and the difference is free of discretisation noise.
        """
        return YieldCurve(self.maturities.copy(), self.zero_rates + bump)
