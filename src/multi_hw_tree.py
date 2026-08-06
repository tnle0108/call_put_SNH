from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .curve import YieldCurve
from .daycount import tenor_to_yearfrac

_M_POS = np.array([[5.0, -4.0, -1.0], [-4.0, 8.0, -4.0], [-1.0, -4.0, 5.0]])
_M_NEG = np.array([[1.0, 4.0, -5.0], [4.0, -8.0, 4.0], [-5.0, 4.0, 1.0]])
REF_CONVENTION = 'actactisda'


def _B(a: float, tau: float) -> float:
    """Hull-White B(t, t+tau) = (1 - e^{-a tau}) / a, a->0 -> tau."""
    if a < 1e-8:
        return tau
    return -np.expm1(-a * tau) / a


def _G(a: float, sigma: float, tau: float) -> float:
    """Cumulative log-bond variance sigma^2 int_0^tau B(u)^2 du (a->0 -> sigma^2 tau^3/3)."""
    if a < 1e-6:
        return sigma**2 * tau**3 / 3.0
    return (sigma**2 / a**2) * (
        tau
        + (2.0 / a) * np.exp(-a * tau)
        - (1.0 / (2.0 * a)) * np.exp(-2.0 * a * tau)
        - 3.0 / (2.0 * a)
    )


def _affine_rate(curve, a: float, sigma: float, t: float, tau: float):
    """Affine coefficients ``(c, slope)`` of the yield ``R(t, t+tau | z) = c + slope*z``.

    The same construction as :meth:`~src.hw_tree.HullWhiteTree._assign_rates`: ``c`` is
    the forward rate less the convexity kernel and ``slope`` is ``B(tau)/tau``.  The
    affine bond price follows as ``P(t, t+tau | z) = exp(-(c + slope*z) * tau)``, so
    this is the single primitive behind both the discount leg and the coupon index --
    only the compounding convention and the tenor differ downstream.
    """
    T = t + tau
    fwd = -np.log(curve.discount(T) / curve.discount(t)) / tau
    convexity = (
        _G(a, sigma, tau) - _G(a, sigma, T) + _G(a, sigma, t)
    ) / (2.0 * tau)
    return fwd - convexity, _B(a, tau) / tau


@dataclass
class CouponDef:
    """One coupon period paid at its map key step (see :class:`BondSpec`).

    ``fixed_rate`` set -> fixed coupon; otherwise floating ``L + margin`` where ``L``
    is the reference rate of tenor ``ref_tenor`` (defaults to ``accrual``) read at the
    paying node. ``floor``/``cap`` are annualized bounds applied as
    ``min(max(rate, floor), cap)``. All rates are annualized; ``accrual`` is the
    year-fraction (e.g. 0.25 quarterly).
    """

    accrual: float
    fixed_rate: float | None = None
    margin: float = 0.0
    ref_tenor: str | None = None
    floor: float | None = None
    cap: float | None = None

    @property
    def is_float(self) -> bool:
        return self.fixed_rate is None


@dataclass
class BondSpec:
    """Face value, redemption step, and coupons keyed by the time step that pays them."""

    face: float
    maturity_step: int
    coupons: dict[int, CouponDef] = field(default_factory=dict)


@dataclass
class ExerciseSpec:
    """Call (issuer) and put (holder) clean strikes keyed by exercise step.

    Bermudan by construction -- exercise is only tested at the listed steps; an
    American window is approximated by listing every step in it. Strikes are compared
    directly to the (dirty) continuation value under the ex-coupon assumption
    (accrued = 0 at coupon-aligned exercise dates).
    """

    call: dict[int, float] = field(default_factory=dict)
    put: dict[int, float] = field(default_factory=dict)


@dataclass
class PricingFlags:
    """Toggle embedded features on/off for twin-run decomposition."""

    floor: bool = True
    cap: bool = True
    call: bool = True
    put: bool = True


@dataclass
class _FactorTree:
    """Hull recombining trinomial geometry for one OU factor on a constant grid."""

    a: float
    sigma: float
    # dt: float
    # n_steps: int
    times: np.ndarray #debug
    times_normal: np.ndarray

    def __post_init__(self) -> None:
        # Hull's band is where branching switches to up/down to keep p in [0, 1]: at
        # j_max the clipped central child leaves m = 1 - 0.184 = 0.816, just inside the
        # |m| <= sqrt(2/3) bound where pm = 2/3 - m^2 turns negative.
        #
        # It diverges as a -> 0 (a=1e-9 asks for j_max = 7.4e8), so cap it at n_steps:
        # from j=0 the lattice cannot reach past level +/-N in N steps, so any wider
        # band is unreachable from the root and costs O(n^2) in pjoint for nothing.
        # Within the reachable cone |j| <= i the clip never binds, so capping leaves
        # every reachable node's probabilities untouched.
        self.dt = np.diff(self.times)
        self.n_steps = len(self.dt)

        # dt_min = self.dt.min()

        # if self.a <= 0:
        #     hull = np.inf
        # else:
        #     hull = np.ceil(0.184 / (self.a * dt_min))

        # self.j_max = max(int(min(hull, self.n_steps)), 1)
        # self.levels = np.arange(-self.j_max, self.j_max + 1)
        # self.n = self.levels.size
        #debug
        self.levels = []
        self.child_idx=[]
        self.central = []
        self.pu=[]
        self.pm=[]
        self.pd=[]
        self.n = []
        self.dx = []

        for i in range(len(self.times)):
            if i == len(self.times)-1:
                dt = self.dt[-1]
            else:
                dt = self.dt[i]

            dx = self.sigma * np.sqrt(3.0 * dt)
            self.dx.append(dx)

            # band width
            if self.a <= 0:
                j_max = self.n_steps
            else:
                hull = np.ceil(0.184/(self.a*dt))
                j_max = int(max(min(hull,self.n_steps),1))

            levels = np.arange(-j_max, j_max+1)

            self.levels.append(levels)
            self.n.append(levels.size)

            # Hull branching
            raw = np.round(levels * (1.0 - self.a * dt)).astype(int)
            
            central = np.clip(raw, -(j_max - 1), j_max - 1)
            self.central.append(central)

            m = (levels*(1-self.a*dt) - central)
            pu = (1.0 + 3.0 * m**2 + 3.0 * m) / 6.0
            pd = (1.0 + 3.0 * m**2 - 3.0 * m) / 6.0
            pm = 1.0 - pu - pd

            self.pu.append(pu)
            self.pm.append(pm)
            self.pd.append(pd)

            offset = j_max

            child = np.stack(
                [
                    central + 1 + offset,
                    central + offset,
                    central - 1 + offset,
                ],
                axis=1,
            )  # shape (n, 3)
            self.child_idx.append(child)

        #debug

        # Central child (absolute level) and (u, m, d) probabilities per node.
        # raw = np.round(self.levels * (1.0 - self.a * self.dt)).astype(int)
        # self.central = np.clip(raw, -(self.j_max - 1), self.j_max - 1)
        # m = self.levels * (1.0 - self.a * self.dt) - self.central
        # self.pu = (1.0 + 3.0 * m**2 + 3.0 * m) / 6.0
        # self.pd = (1.0 + 3.0 * m**2 - 3.0 * m) / 6.0
        # self.pm = 1.0 - self.pu - self.pd

        # # Child indices (into levels) for the up/mid/down branches of each node.
        # off = self.j_max
        # self.child_idx = np.stack(
        #     [
        #         self.central + 1 + off,
        #         self.central + off,
        #         self.central - 1 + off,
        #     ],
        #     axis=1,
        # )  # shape (n, 3)

    # def x(self, j_level: int) -> float:
    #     return j_level * self.dx
    def x(self, step, level):
        return (level *  self.dx[step])

@dataclass
class MultiCurveHWTree:
    """Two-factor multi-curve Hull-White Normal tree.

    Parameters
    ----------
    a_r, sigma_r:
        Mean reversion / volatility of the discount factor ``x``.
    a_L, sigma_L:
        Mean reversion / volatility of the reference factor ``y``.
    rho:
        Instantaneous correlation of the two Brownian motions.
    disc_curve, ref_curve:
        Discount curve ``P^r(0,t)`` and reference curve ``P^L(0,t)``.
    dt, n_steps:
        Constant step and number of steps; ``times = arange(n_steps+1) * dt``. Choose
        ``dt`` so coupon / call / put / maturity dates land on nodes.
    """

    a_r: float
    sigma_r: float
    a_L: float
    sigma_L: float
    rho: float
    disc_curve: YieldCurve
    ref_curve: YieldCurve
    disc_convention: str
    ref_convention: str
    # dt: float
    # n_steps: int
    times: np.ndarray #debug
    times_normal: np.ndarray

    def __post_init__(self) -> None:
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        # self.times = np.arange(self.n_steps + 1) * self.dt
        self.times=np.asarray(self.times)
        if self.times[0]!=0:
            raise ValueError("Times must start at 0")
        self.n_steps=len(self.times)-1 #debug
        if self.n_steps < 1:
            raise ValueError("n_steps >= 1 required")
        # self.dt_steps=np.diff(self.times) 
        # self.xt = _FactorTree(self.a_r, self.sigma_r, self.dt, self.n_steps)
        self.xt = _FactorTree(self.a_r, self.sigma_r, self.times, self.times_normal, self.disc_convention) #debug
        # self.yt = _FactorTree(self.a_L, self.sigma_L, self.dt, self.n_steps)
        self.yt = _FactorTree(self.a_L, self.sigma_L, self.times, self.times_normal) #debug
        self._build_joint_probs()
        self._assign_rates()

    # def _build_joint_probs(self) -> None:
    #     nx, ny = self.xt.n, self.yt.n
    #     px = np.stack([self.xt.pu, self.xt.pm, self.xt.pd], axis=1)
    #     py = np.stack([self.yt.pu, self.yt.pm, self.yt.pd], axis=1)
    #     eps = self.rho / 36.0
    #     M = _M_POS if self.rho >= 0 else _M_NEG
    #     shift = eps * M

    #     self.pjoint = np.empty((nx, ny, 3, 3))
    #     for jx in range(nx):
    #         for ky in range(ny):
    #             p0 = np.outer(px[jx], py[ky])
    #             lam = 1.0
    #             neg = shift < 0
    #             if np.any(neg):
    #                 lam = min(1.0, np.min(p0[neg] / -shift[neg]))
    #             p = p0 + lam * shift
    #             p = np.clip(p, 0.0, 1.0)
    #             p /= p.sum()
    #             self.pjoint[jx, ky] = p
    def _build_joint_probs(self):
        self.pjoint=[]
        for i in range(self.n_steps):
            px = np.stack([self.xt.pu[i], self.xt.pm[i], self.xt.pd[i]],axis=1)
            py = np.stack([self.yt.pu[i], self.yt.pm[i], self.yt.pd[i]],axis=1)
            nx = len(px)
            ny = len(py)
            step_prob = np.empty((nx,ny,3,3))
            eps = self.rho/36
            M = (_M_POS if self.rho>=0 else _M_NEG)
            shift = eps*M

            for jx in range(nx):
                for ky in range(ny):
                    p0 = np.outer(px[jx], py[ky])
                    lam = 1
                    neg = shift<0
                    if np.any(neg):
                        lam=min(1, np.min(p0[neg]/(-shift[neg])))
                    p = p0 + lam * shift
                    p = np.clip(p, 0, 1)
                    p /= p.sum()
                    step_prob[jx,ky] = p

            self.pjoint.append(step_prob)

    def _assign_rates(self) -> None:
        """Affine node rates for the discount leg, ``r_i(x) = c[i] + slope[i] * x``.

        The two-factor analogue of :meth:`~src.hw_tree.HullWhiteTree._assign_rates`,
        specialized to the constant grid.  The rollback consumes the one-step discount
        ``step_discount[i] = exp(-r[i] * dt)`` rather than the rate itself.
        """
        N = self.n_steps
        self.c = []
        self.slope = []
        self.r = []
        self.step_discount = []

        for i in range(N):
            dt = self.times[i+1] - self.times[i]
            c,slope = _affine_rate(
                self.disc_curve,
                self.a_r,
                self.sigma_r,
                self.times[i],
                dt
            )
            self.c.append(c)
            self.slope.append(slope)
            x_values = (self.xt.levels[i] * self.xt.dx[i])
            rates = (c + slope*x_values)
            self.r.append(rates)
            self.step_discount.append(np.exp( - rates * dt))

        # x_vals = self.xt.levels * self.xt.dx
        # self.c = np.zeros(N)
        # self.slope = np.zeros(N)
        # self.r = [None] * N
        # self.step_discount = []

        # for i in range(N):
        #     self.c[i], self.slope[i] = _affine_rate(
        #         self.disc_curve, self.a_r, self.sigma_r, self.times[i], self.dt
        #     )
        #     self.r[i] = self.c[i] + self.slope[i] * x_vals
        #     self.step_discount.append(np.exp(-self.r[i] * self.dt))

    def short_rates(self, i: int):
        """(levels, period rates) at step ``i``; mirrors ``HullWhiteTree.short_rates``."""
        return self.xt.levels, self.r[i]

    def reference_rate(self, i: int, tenor: float):
        """Reference rate L_i(tenor | y) over the y-levels at step ``i`` (simple comp).

        Built from the same affine ``c + slope * y`` form as the discount leg, but off
        ``ref_curve``/``y`` and at an arbitrary ``tenor`` (which need not equal ``dt``).
        The coupon index is quoted simple-compounded, so the continuously-compounded
        yield ``R`` is converted as ``L = (e^{R*tenor} - 1) / tenor``.
        """
        c, slope = _affine_rate(
            self.ref_curve, self.a_L, self.sigma_L, self.times[i], tenor
        )
        R = c + slope * (self.yt.levels[i]*self.yt.dx[i])
        return np.expm1(R * tenor) / tenor

    def _coupon_amounts(self, i: int, bond: BondSpec, flags: PricingFlags):
        """Coupon cash paid at step ``i``: array over y-levels, or 0.0 if none."""
        cdef = bond.coupons.get(i)
        if cdef is None:
            return 0.0
        if cdef.is_float:
            if cdef.ref_tenor is not None:
                ref_tenor_year_frac = tenor_to_yearfrac(cdef.ref_tenor, REF_CONVENTION,self.times_normal[i])
            else:
                ref_tenor_year_frac = cdef.accrual
            tenor = ref_tenor_year_frac
            rate = self.reference_rate(i, tenor) + cdef.margin
        else:
            rate = np.full(self.yt.n[i],cdef.fixed_rate)
        if flags.floor and cdef.floor is not None:
            rate = np.maximum(rate, cdef.floor)
        if flags.cap and cdef.cap is not None:
            rate = np.minimum(rate, cdef.cap)
        return rate * cdef.accrual * bond.face

    def price(
        self,
        bond: BondSpec,
        exercise: ExerciseSpec | None = None,
        flags: PricingFlags | None = None,
    ) -> float:
        """Present value today (dirty) of the bond via one backward induction."""
        flags = flags or PricingFlags()
        N = bond.maturity_step
        print(N)
        if N > self.n_steps:
            raise ValueError("bond.maturity_step exceeds n_steps")
        nx=self.xt.n[N]
        ny=self.yt.n[N]
        # nx, ny = self.xt.n, self.yt.n
        # xc, yc = self.xt.child_idx, self.yt.child_idx  # (nx,3), (ny,3)

        cpn_N = self._coupon_amounts(N, bond, flags)
        V = bond.face + np.zeros((nx, ny)) + np.broadcast_to(cpn_N, (nx, ny))

        for i in range(N - 1, -1, -1):
            xc=self.xt.child_idx[i]
            yc=self.yt.child_idx[i]
            # prob=self.pjoint[i]

            D_i = self.step_discount[i]

            nx=len(self.xt.levels[i])
            ny=len(self.yt.levels[i])

            cont=np.zeros((nx,ny))

            xc=self.xt.child_idx[i]
            yc=self.yt.child_idx[i]

            for a in range(3):
                xi=xc[:,a]
                for b in range(3):
                    yi=yc[:,b]

                    cont += (self.pjoint[i][:,:,a,b] * V[np.ix_(xi,yi)])

            cont *= D_i[:,None]


            # D_i = self.step_discount[i]  # (nx,) depends on x only
            # cont = np.zeros((nx, ny))
            # for a in range(3):
            #     xi = xc[:, a]  # (nx,) child x-index per node
            #     for b in range(3):
            #         yi = yc[:, b]  # (ny,)
            #         cont += self.pjoint[:, :, a, b] * V[np.ix_(xi, yi)]
            # cont *= D_i[:, None]

            if exercise is not None:
                if flags.call and i in exercise.call:
                    cont = np.minimum(exercise.call[i], cont)
                if flags.put and i in exercise.put:
                    cont = np.maximum(exercise.put[i], cont)

            cpn = self._coupon_amounts(i, bond, flags)
            V = cont + np.broadcast_to(cpn, (nx, ny))
            root_x=len(self.xt.levels[0])//2
            root_y=len(self.yt.levels[0])//2

        return float(V[root_x,root_y])  # root: x=0, y=0

    def decompose(
        self, bond: BondSpec, exercise: ExerciseSpec | None = None
    ) -> dict:
        """Isolate each embedded-option value by toggling flags (section 8)."""
        off = PricingFlags(floor=False, cap=False, call=False, put=False)
        r0 = self.price(bond, exercise, off)
        full = self.price(bond, exercise, PricingFlags())

        def one(**kw):
            return self.price(
                bond,
                exercise,
                PricingFlags(
                    **{
                        **dict(floor=False, cap=False, call=False, put=False),
                        **kw,
                    }
                ),
            )

        v_floor = one(floor=True) - r0
        v_cap = r0 - one(cap=True)
        v_call = r0 - one(call=True)
        v_put = one(put=True) - r0
        interaction = full - (r0 + v_floor - v_cap - v_call + v_put)
        return {
            "straight": r0,
            "full": full,
            "floor": v_floor,
            "cap": v_cap,
            "call": v_call,
            "put": v_put,
            "interaction": interaction,
        }

    def realized_correlation(self, jx: int, ky: int) -> float:
        """Realized branch correlation at node index (jx, ky); should equal rho interior."""
        p = self.pjoint[jx, ky]
        dx = np.array([1.0, 0.0, -1.0]) * self.xt.dx
        dy = np.array([1.0, 0.0, -1.0]) * self.yt.dx
        ex = (p.sum(axis=1) * dx).sum()
        ey = (p.sum(axis=0) * dy).sum()
        cov = sum(
            p[a, b] * (dx[a] - ex) * (dy[b] - ey)
            for a in range(3)
            for b in range(3)
        )
        vx = (p.sum(axis=1) * (dx - ex) ** 2).sum()
        vy = (p.sum(axis=0) * (dy - ey) ** 2).sum()
        return cov / np.sqrt(vx * vy)
