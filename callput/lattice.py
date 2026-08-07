"""Trinomial lattice geometry and branch probabilities on a non-uniform grid.

Geometry follows :class:`src.hw_tree.HullWhiteTree`: ``dx`` is rebuilt for every
time level, ``dx_i = sqrt(3 V_i)``.  That is what keeps ``V_i / dx_i**2 = 1/3``
whatever the step lengths are, so the probabilities stay in Hull's standard
``1/6, 2/3, 1/6`` form and are well conditioned even where the grid is uneven.
The price is that the meaning of a level index changes along the tree -- between
two steps it rescales by ``sqrt(dt_i / dt_{i+1})`` -- so each time level keeps its
own ``j_min`` / ``j_max`` and the value array changes width as the rollback walks.

The joint probabilities follow :func:`src.multi_hw_tree.MultiCurveHWTree._build_joint_probs`
(same correlation matrices and lambda scaling) but vectorised, and cached by the
pair of adjacent step lengths in days rather than rebuilt per step.
"""

from __future__ import annotations

import numpy as np

from .leg import CurveLeg, ou_variance

_SQRT3 = np.sqrt(3.0)

# Correlation adjustment applied to the outer product of the two marginals.
_M_POS = np.array([[5.0, -4.0, -1.0], [-4.0, 8.0, -4.0], [-1.0, -4.0, 5.0]])
_M_NEG = np.array([[1.0, 4.0, -5.0], [4.0, -8.0, 4.0], [-5.0, 4.0, 1.0]])


class FactorLattice:
    """Geometry and branch probabilities for one Hull-White factor.

    Attributes per time level ``i``: ``j_min[i]``, ``j_max[i]``, ``dx[i]``,
    ``levels[i]``, ``x[i]``, ``n[i]``.  Per step ``i``: ``prob[i]`` of shape
    ``(n[i], 3)`` and ``child[i]`` of shape ``(n[i], 3)`` holding indices into the
    level array of ``i + 1`` (up / mid / down).
    """

    def __init__(self, leg: CurveLeg, days: np.ndarray, n_sigma: float = 6.0) -> None:
        self.leg = leg
        self.days = np.asarray(days, dtype=int)
        self.step_days = np.diff(self.days)
        n = self.step_days.size
        if n < 1:
            raise ValueError("need at least one step")

        self.dt = self.step_days / leg.denom
        self.var = np.array([ou_variance(leg.a, leg.sigma, dt) for dt in self.dt])

        # dx at time level i is set by the step *arriving* at i; level 0 is a
        # single node so its spacing is unused.
        self.dx = np.zeros(n + 1)
        self.dx[1:] = np.sqrt(3.0 * self.var)

        self.trunc = self._truncation(n_sigma)
        self._build()

    # -- truncation ---------------------------------------------------------
    def _truncation(self, n_sigma: float) -> np.ndarray:
        """Widest level kept at each time level.

        The state space is truncated at **one half-width in x**, not at a level
        count chosen per step.  Levels then follow as ``ceil(x_max / dx_i)``, which
        keeps the boundary at the same place in state space however the spacing
        changes, so a step onto a finer spacing maps the outermost level onto the
        new outermost level and the central-child clip binds only there.

        Choosing per step instead is what goes wrong: ``n_sigma`` standard
        deviations scales as ``1/dx ~ 1/sqrt(dt)`` while Hull's ``0.184/(a dt)``
        scales as ``1/dt``.  When one bound binds on the short steps and the other
        on the long ones, the level count jumps between adjacent times and a whole
        band of levels is clipped at once -- far more than the degenerate-branch
        fallback is meant to absorb.

        Both bounds are still honoured, just converted to x first.  ``n_sigma``
        sigmas of the terminal distribution normally binds and is far inside
        Hull's band on a fine grid: at a 7-day step with a = 5% it keeps ~63 levels
        against Hull's ~192, shrinking a two-factor grid ninefold for a tail
        carrying probability ~1e-9.  Hull's band is kept as the outer limit because
        it is exactly the distance at which a clipped central branch still
        satisfies ``pm >= 0``, so a caller who sets ``n_sigma`` very large degrades
        to the classical degeneracy-free tree rather than to a memory blow-up.

        The terminal std is used at every level rather than the running one: the
        cone growing out of the root already keeps early levels narrow, and a
        time-varying bound would make the geometry depend on more than the adjacent
        step lengths, defeating the joint-probability cache.
        """
        x_max = n_sigma * self.leg.terminal_std(int(self.days[-1]))
        if self.leg.a > 1e-8:
            # Hull's band in x units; the longest step gives the tightest limit.
            x_max = min(
                x_max,
                min(
                    0.184 * self.dx[i] / (self.leg.a * self.dt[i - 1])
                    for i in range(1, self.dx.size)
                ),
            )
        out = np.zeros(self.dx.size, dtype=int)
        out[1:] = np.maximum(1, np.ceil(x_max / self.dx[1:]).astype(int))
        return out

    # -- build --------------------------------------------------------------
    def _build(self) -> None:
        n = self.step_days.size
        self.j_min = np.zeros(n + 1, dtype=int)
        self.j_max = np.zeros(n + 1, dtype=int)
        self.prob: list[np.ndarray] = []
        self.child: list[np.ndarray] = []
        self.n_degenerate = 0

        for i in range(n):
            levels = np.arange(self.j_min[i], self.j_max[i] + 1)
            p, k, bad = self.transition(levels, i)

            self.j_min[i + 1] = int(k.min()) - 1
            self.j_max[i + 1] = int(k.max()) + 1
            self.prob.append(p)
            self.child.append(
                np.stack([k + 1, k, k - 1], axis=1) - self.j_min[i + 1]
            )

            if bad.any():
                self.n_degenerate += int(bad.sum())
                outer = (levels <= self.j_min[i] + 1) | (levels >= self.j_max[i] - 1)
                if not np.all(outer[bad]):
                    raise ValueError(
                        f"branch probabilities went negative away from the "
                        f"boundary at step {i}; the state space is truncated too "
                        f"tightly -- raise n_sigma"
                    )

        self.levels = [
            np.arange(self.j_min[i], self.j_max[i] + 1) for i in range(n + 1)
        ]
        self.x = [lv * self.dx[i] for i, lv in enumerate(self.levels)]
        self.n = np.array([lv.size for lv in self.levels], dtype=int)

    def transition(self, levels: np.ndarray, i: int):
        """``(prob, central_level, degenerate)`` for ``levels`` over step ``i``.

        Kept public and level-agnostic so the joint-probability cache can evaluate
        it once on a superset range and slice, instead of per step.
        """
        var_i = self.var[i]
        dx_next = self.dx[i + 1]
        trunc_next = self.trunc[i + 1]

        mean = levels * self.dx[i] * np.exp(-self.leg.a * self.dt[i])
        k = np.clip(
            np.rint(mean / dx_next).astype(int),
            -(trunc_next - 1),
            trunc_next - 1,
        )
        eta = mean - k * dx_next

        sd = np.sqrt(var_i)
        pu = 1.0 / 6.0 + eta**2 / (6.0 * var_i) + eta / (2.0 * _SQRT3 * sd)
        pm = 2.0 / 3.0 - eta**2 / (3.0 * var_i)
        pd = 1.0 / 6.0 + eta**2 / (6.0 * var_i) - eta / (2.0 * _SQRT3 * sd)
        p = np.stack([pu, pm, pd], axis=1)

        # Clipping the central child at the truncation boundary pushes |eta|
        # past dx/2, where moment matching can no longer hold.  Those nodes carry
        # probability ~1e-9, so they collapse to a single deterministic branch.
        bad = p.min(axis=1) < 0.0
        if bad.any():
            p[bad] = np.array([0.0, 1.0, 0.0])
        return p, k, bad


class Lattice:
    """One or two factors sharing a time grid.

    One factor is the single-curve engine (fixed-rate bonds); two factors add the
    reference index with correlation ``rho``.  The rollback below is written on
    negative axes so an optional leading fixing axis rides along untouched.
    """

    def __init__(
        self,
        legs: list[CurveLeg],
        days: np.ndarray,
        rho: float = 0.0,
        n_sigma: float = 6.0,
    ) -> None:
        if not 1 <= len(legs) <= 2:
            raise ValueError("Lattice takes one or two factors")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        self.days = np.asarray(days, dtype=int)
        self.step_days = np.diff(self.days)
        self.rho = float(rho)
        self.factors = [FactorLattice(leg, self.days, n_sigma) for leg in legs]
        self._joint: dict[tuple[int, int], np.ndarray] = {}
        self._sup = [int(f.trunc.max()) for f in self.factors]

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def n_steps(self) -> int:
        return int(self.step_days.size)

    def shape(self, i: int) -> tuple[int, ...]:
        """State shape of the value array at time level ``i``."""
        return tuple(int(f.n[i]) for f in self.factors)

    # -- joint probabilities ------------------------------------------------
    def _pair_key(self, i: int) -> tuple[int, int]:
        """Everything about step ``i`` is fixed by the incoming and outgoing spans.

        ``dx[i]`` comes from step ``i-1`` and ``dx[i+1]``, ``dt[i]``, ``trunc[i+1]``
        all come from step ``i``.  With a ~7-day grid there are two or three
        distinct spans, hence at most nine cached joints instead of one per step.
        """
        prev = int(self.step_days[i - 1]) if i > 0 else 0
        return (prev, int(self.step_days[i]))

    def joint(self, i: int) -> np.ndarray:
        """``(n_x, n_y, 3, 3)`` joint branch probabilities at step ``i``."""
        key = self._pair_key(i)
        full = self._joint.get(key)
        if full is None:
            fx, fy = self.factors
            lx = np.arange(-self._sup[0], self._sup[0] + 1)
            ly = np.arange(-self._sup[1], self._sup[1] + 1)
            px, _, _ = fx.transition(lx, i)
            py, _, _ = fy.transition(ly, i)
            full = _combine(px, py, self.rho)
            self._joint[key] = full

        fx, fy = self.factors
        ox, oy = self._sup
        return full[
            fx.j_min[i] + ox : fx.j_max[i] + ox + 1,
            fy.j_min[i] + oy : fy.j_max[i] + oy + 1,
        ]

    # -- rollback -----------------------------------------------------------
    def rollback(self, V: np.ndarray, i: int, step_discount: np.ndarray) -> np.ndarray:
        """Value at time level ``i+1`` -> continuation value at time level ``i``.

        A leading fixing axis, if present, is untouched: it does not branch and
        carries no probability of its own, so the same probabilities apply
        slice by slice.
        """
        if self.n_factors == 1:
            f = self.factors[0]
            p, c = f.prob[i], f.child[i]
            cont = p[:, 0] * V.take(c[:, 0], axis=-1)
            cont += p[:, 1] * V.take(c[:, 1], axis=-1)
            cont += p[:, 2] * V.take(c[:, 2], axis=-1)
            return cont * step_discount

        fx, fy = self.factors
        pj = self.joint(i)
        cx, cy = fx.child[i], fy.child[i]
        cont = None
        for a in range(3):
            va = V.take(cx[:, a], axis=-2)
            for b in range(3):
                term = pj[..., a, b] * va.take(cy[:, b], axis=-1)
                cont = term if cont is None else cont + term
        return cont * step_discount[:, None]

    # -- diagnostics --------------------------------------------------------
    def realized_correlation(self, i: int, jx: int = 0, ky: int = 0) -> float:
        """Branch correlation at an interior node of step ``i``; should equal rho."""
        if self.n_factors != 2:
            raise ValueError("correlation needs two factors")
        fx, fy = self.factors
        p = self.joint(i)[jx - fx.j_min[i], ky - fy.j_min[i]]
        dx = np.array([1.0, 0.0, -1.0]) * fx.dx[i + 1]
        dy = np.array([1.0, 0.0, -1.0]) * fy.dx[i + 1]
        ex = (p.sum(axis=1) * dx).sum()
        ey = (p.sum(axis=0) * dy).sum()
        cov = sum(
            p[a, b] * (dx[a] - ex) * (dy[b] - ey) for a in range(3) for b in range(3)
        )
        vx = (p.sum(axis=1) * (dx - ex) ** 2).sum()
        vy = (p.sum(axis=0) * (dy - ey) ** 2).sum()
        return float(cov / np.sqrt(vx * vy))

    @property
    def n_degenerate(self) -> int:
        return sum(f.n_degenerate for f in self.factors)

    @property
    def max_nodes(self) -> int:
        sizes = [int(np.prod([f.n[i] for f in self.factors])) for i in range(len(self.days))]
        return max(sizes)


def _combine(px: np.ndarray, py: np.ndarray, rho: float) -> np.ndarray:
    """Correlate two marginals into a 3x3 joint per node pair.

    Same construction as ``MultiCurveHWTree._build_joint_probs``: the outer
    product is nudged by ``(rho / 36) * M``, scaled back by ``lambda`` wherever the
    nudge would drive an entry negative, then clipped and renormalised.
    """
    eps = rho / 36.0
    shift = eps * (_M_POS if rho >= 0 else _M_NEG)
    p0 = px[:, None, :, None] * py[None, :, None, :]

    neg = shift < 0.0
    if neg.any():
        denom = np.where(neg, -shift, 1.0)
        ratio = np.where(neg, p0 / denom, np.inf)
        lam = np.minimum(1.0, ratio.min(axis=(2, 3)))
    else:
        lam = np.ones(p0.shape[:2])

    p = p0 + lam[:, :, None, None] * shift
    np.clip(p, 0.0, None, out=p)
    p /= p.sum(axis=(2, 3), keepdims=True)
    return p
