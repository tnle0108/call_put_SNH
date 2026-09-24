from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .curve import YieldCurve

_SQRT3 = np.sqrt(3.0)


@dataclass
class HullWhiteTree:
    """Hull-White trinomial tree with analytic (affine) node rates.

    Parameters
    ----------
    a:
        Mean-reversion speed (>= 0).
    sigma:
        Short-rate volatility (> 0).
    times:
        Node times in years, starting at 0 and strictly increasing, e.g.
        ``np.linspace(0, 3, 7)`` or a schedule-aligned grid.
    curve:
        Market :class:`~src.curve.YieldCurve` supplying ``P(0, t)``.

    After construction the tree exposes, per time step ``i``:
    ``j_min[i] .. j_max[i]`` (node index range), ``x[i]``/``r[i]`` (state values and
    period rates), ``c[i]``/``slope[i]`` (affine node-rate coefficients),
    ``central[i]`` (central child index per node) and ``pu/pm/pd[i]`` (branch
    probabilities).
    """

    a: float
    sigma: float
    times: np.ndarray
    curve: YieldCurve

    # -- built state (filled by _build) ------------------------------------
    dt: np.ndarray = field(init=False)
    dx: np.ndarray = field(init=False)
    j_min: np.ndarray = field(init=False)
    j_max: np.ndarray = field(init=False)
    x: list = field(init=False)
    r: list = field(init=False)
    c: np.ndarray = field(init=False)
    slope: np.ndarray = field(init=False)
    central: list = field(init=False)
    pu: list = field(init=False)
    pm: list = field(init=False)
    pd: list = field(init=False)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1 or self.times.size < 2:
            raise ValueError(
                "times must be a 1-D grid with at least two points"
            )
        if self.times[0] != 0.0:
            raise ValueError("times must start at 0")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")
        if self.a < 0:
            raise ValueError("mean reversion a must be >= 0")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        self._build_stage1()
        self._assign_rates()

    def _ou_variance(self, dt: float) -> float:
        """Conditional variance sigma^2/(2a) (1 - e^{-2a dt}), a->0 -> sigma^2 dt."""
        if self.a < 1e-8:
            return self.sigma**2 * dt
        return self.sigma**2 * (-np.expm1(-2.0 * self.a * dt)) / (2.0 * self.a)

    def _B(self, tau: float) -> float:
        """Hull-White B(t, t+tau) = (1 - e^{-a tau}) / a, a->0 -> tau."""
        if self.a < 1e-8:
            return tau
        return -np.expm1(-self.a * tau) / self.a

    def _integrated_variance(self, tau: float) -> float:
        """IV(tau) = sigma^2 int_0^tau B(u)^2 du, the affine-bond convexity kernel.

        Closed form sigma^2/a^2 [tau + (2/a)e^{-a tau} - (1/2a)e^{-2a tau} - 3/2a];
        as a->0 this tends to sigma^2 tau^3 / 3.
        """
        a, s = self.a, self.sigma
        if a < 1e-6:
            return s**2 * tau**3 / 3.0
        return (s**2 / a**2) * (
            tau
            + (2.0 / a) * np.exp(-a * tau)
            - (1.0 / (2.0 * a)) * np.exp(-2.0 * a * tau)
            - 3.0 / (2.0 * a)
        )

    @property
    def n_steps(self) -> int:
        return self.times.size - 1

    # -- Stage 1: geometry & probabilities of the x-tree -------------------
    def _build_stage1(self) -> None:
        N = self.n_steps
        self.dt = np.diff(self.times)
        self.dx = np.zeros(N + 1)
        for i in range(1, N + 1):
            self.dx[i] = np.sqrt(3.0 * self._ou_variance(self.dt[i - 1]))

        self.j_min = np.zeros(N + 1, dtype=int)
        self.j_max = np.zeros(N + 1, dtype=int)
        self.x = [np.zeros(1)]  # x[0] = {0}
        self.central, self.pu, self.pm, self.pd = [], [], [], []

        for i in range(N):
            Vi2 = self._ou_variance(self.dt[i])
            Vi = np.sqrt(Vi2)
            decay = np.exp(-self.a * self.dt[i])
            dxn = self.dx[i + 1]

            js = np.arange(self.j_min[i], self.j_max[i] + 1)
            x_i = js * self.dx[i]
            M = x_i * decay
            k = np.round(M / dxn).astype(int)
            eta = M - k * dxn

            pu = 1.0 / 6.0 + eta**2 / (6.0 * Vi2) + eta / (2.0 * _SQRT3 * Vi)
            pm = 2.0 / 3.0 - eta**2 / (3.0 * Vi2)
            pd = 1.0 / 6.0 + eta**2 / (6.0 * Vi2) - eta / (2.0 * _SQRT3 * Vi)

            self.central.append(k)
            self.pu.append(pu)
            self.pm.append(pm)
            self.pd.append(pd)

            self.j_min[i + 1] = int(k.min()) - 1
            self.j_max[i + 1] = int(k.max()) + 1
            js_next = np.arange(self.j_min[i + 1], self.j_max[i + 1] + 1)
            self.x.append(js_next * self.dx[i + 1])

    # -- Stage 2: analytic (affine) node rates -----------------------------
    def _assign_rates(self) -> None:
        N = self.n_steps
        self.c = np.zeros(N)
        self.slope = np.zeros(N)
        self.r = [None] * N

        for i in range(N):
            tau = self.dt[i]
            B_i = self._B(tau)
            iv_step = self._integrated_variance(tau)
            iv_end = self._integrated_variance(self.times[i + 1])
            iv_start = self._integrated_variance(self.times[i])

            fwd = (
                -np.log(
                    self.curve.discount(self.times[i + 1])
                    / self.curve.discount(self.times[i])
                )
                / tau
            )
            convexity = (iv_step - iv_end + iv_start) / (2.0 * tau)

            self.c[i] = fwd - convexity
            self.slope[i] = B_i / tau

            js = np.arange(self.j_min[i], self.j_max[i] + 1)
            self.r[i] = self.c[i] + self.slope[i] * (js * self.dx[i])

    def short_rates(self, i: int):
        """(j indices, period rates) at time step ``i`` (0 <= i < n_steps)."""
        js = np.arange(self.j_min[i], self.j_max[i] + 1)
        return js, self.r[i]

    def zero_coupon_price(self, m: int) -> float:
        """Tree price today of a zero-coupon bond maturing at ``times[m]`` via rollback.

        With analytic node rates this reproduces ``curve.discount(times[m])`` only up
        to the tree's discretization error (see :meth:`curve_repricing_error`).
        """
        if not 0 <= m <= self.n_steps:
            raise ValueError("m out of range")
        value = np.ones(self.j_max[m] - self.j_min[m] + 1)
        for i in range(m - 1, -1, -1):
            off = self.j_min[i + 1]
            k = self.central[i]
            up = value[k + 1 - off]
            mid = value[k - off]
            dn = value[k - 1 - off]
            expected = self.pu[i] * up + self.pm[i] * mid + self.pd[i] * dn
            value = np.exp(-self.r[i] * self.dt[i]) * expected
        return float(value[0])

    def curve_repricing_error(self) -> float:
        """Max absolute discount-factor error of the tree vs the market curve.

        Zero for the analytic rates would require the tree to integrate the full
        Gaussian law exactly; in practice this is a small residual that shrinks as
        the grid refines.
        """
        return max(
            abs(self.zero_coupon_price(m) - self.curve.discount(self.times[m]))
            for m in range(1, self.n_steps + 1)
        )
