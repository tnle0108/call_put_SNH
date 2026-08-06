from functools import cached_property
import numpy as np
from quantmr.config import load_spec
from quantmr.curve.curvenode import CurveNode, find_date_with_lookback
from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.utils import Compound, Interp, align_arr


class ZeroCurve:
    ZEROCURVE_CACHE: dict[tuple, "ZeroCurve"] = {}

    def __init__(
        self,
        curvenode: CurveNode,
        as_of_idx: int = 0,
        as_of_override: np.datetime64 | None = None,
        **kwargs,
    ):
        self.curvenode = curvenode
        self._as_of_idx = as_of_idx
        self._as_of_override = as_of_override
        self.kwargs = kwargs

    @property
    def as_of(self) -> np.datetime64:
        if self._as_of_override is not None:
            return self._as_of_override
        return self.curvenode.as_of[self._as_of_idx]

    @property
    def tau(self) -> np.ndarray:
        return self.curvenode.node_tau_at(self._as_of_idx)

    @cached_property
    def interp(self) -> Interp:
        cc_rate = Compound.get("continuous").rate_from_df(
            self.curvenode.node_df_at(self._as_of_idx), self.tau
        )
        return Interp(
            x=self.tau, y=cc_rate, kind="linear", fill_value="extrapolate"
        )

    def df(
        self, t: float | np.float64 | str | np.datetime64 | list | np.ndarray
    ) -> np.ndarray:
        t = self.curvenode._norm_t(t, as_of=self.as_of)
        cc_rate = self.interp.interpolate(t)
        return Compound.get("continuous").df_from_rate(cc_rate, t)

    def rate(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        compounding: str | None = None,
    ) -> np.ndarray:
        t = self.curvenode._norm_t(t, as_of=self.as_of)
        return Compound.get(compounding).rate_from_df(self.df(t), t)

    def forward_df(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        T: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        **align_arr_kwargs,
    ) -> np.ndarray:
        t = self.curvenode._norm_t(t, as_of=self.as_of)
        T = self.curvenode._norm_t(T, as_of=self.as_of)
        try:
            t, T = align_arr(t, T, **align_arr_kwargs)
        except ValueError as e:
            raise ValueError(
                "t and T must be broadcastable to the same shape"
            ) from e
        df_t = self.df(t)
        df_T = self.df(T)
        return np.divide(
            df_T, df_t, out=np.full_like(df_T, np.nan), where=df_t != 0
        )

    def forward_rate(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        T: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        compounding: str | None = None,
        dt: float | np.float64 = 1e-6,
        **align_arr_kwargs,
    ) -> np.ndarray:
        t = self.curvenode._norm_t(t, as_of=self.as_of)
        T = self.curvenode._norm_t(T, as_of=self.as_of)
        try:
            t, T = align_arr(t, T, **align_arr_kwargs)
        except ValueError as e:
            raise ValueError(
                "t and T must be broadcastable to the same shape"
            ) from e
        T = np.where(np.isclose(t, T), T + dt, T)
        fwd_df = self.forward_df(t, T)
        return Compound.get(compounding).rate_from_df(fwd_df, T - t)

    @classmethod
    def get(
        cls,
        curve_name: str,
        as_of: str | np.datetime64,
        lookback: int = 10,
        selected_col: list[str] | slice | None = None,
    ) -> "ZeroCurve":
        curve_name = curve_name.lower()
        as_of_d = np.datetime64(as_of).astype("datetime64[D]")
        cache_key = (curve_name, as_of_d)
        if cache_key in cls.ZEROCURVE_CACHE:
            return cls.ZEROCURVE_CACHE[cache_key]

        curvenode = CurveNode.get(curve_name, selected_col=selected_col)

        found_date = find_date_with_lookback(as_of_d, curvenode.as_of, lookback)
        if found_date is None:
            raise ValueError(
                f"Curve as_of date {as_of_d} "
                f"not found in curve data for {curve_name} "
                f"(lookback={lookback} days)"
            )
        idx = curvenode.date_index(found_date)

        as_of_override = as_of_d if found_date != as_of_d else None

        curve_spec = load_spec("curve").get(curve_name, {})
        zc = cls(
            curvenode,
            as_of_idx=idx,
            as_of_override=as_of_override,
            **curve_spec,
        )
        cls.ZEROCURVE_CACHE[cache_key] = zc
        return zc


class SimZeroCurve:
    SIMZEROCURVE_CACHE: dict[tuple, "SimZeroCurve"] = {}

    def __init__(
        self,
        zerocurve: ZeroCurve,
        sim: str | np.datetime64 | None = None,
        model: HullWhite | None = None,
        epsilon: float | np.float64 | list | np.ndarray | None = None,
        **kwargs,
    ):
        self.zerocurve = zerocurve
        self.curvenode = zerocurve.curvenode
        self._sim = sim
        self.model = model or HullWhite()
        self._epsilon = epsilon
        self.kwargs = kwargs

    @property
    def as_of(self) -> np.datetime64:
        return self.zerocurve.as_of

    @property
    def sim(self) -> np.datetime64:
        sim = np.datetime64(self._sim or self.as_of).astype("datetime64[D]")
        if sim < self.as_of:
            raise ValueError("Sim date cannot be before as_of date")
        return sim

    @cached_property
    def t_sim(self) -> np.float64:
        return np.float64(
            self.curvenode.daycount.yearfrac(self.as_of, self.sim)
        )

    @cached_property
    def tau(self) -> np.ndarray:
        return self.t_sim + self.zerocurve.tau

    @property
    def epsilon(self) -> np.ndarray:
        return np.atleast_1d(
            np.asarray(
                self._epsilon if self._epsilon is not None else 0.0,
                dtype=np.float64,
            )
        ).ravel()

    @epsilon.setter
    def epsilon(
        self, epsilon: float | np.float64 | list | np.ndarray | None
    ) -> None:
        if epsilon is self._epsilon:
            return
        self._epsilon = epsilon
        if "_knots" in self.__dict__:
            del self.__dict__["_knots"]

    @cached_property
    def _knots(self) -> tuple[np.ndarray, np.ndarray]:
        fwd_rate = self.zerocurve.forward_rate(
            self.t_sim, self.tau, compounding="continuous"
        )
        delta_t = self.tau - self.t_sim
        adj = self.model.rate_adjustment(
            self.t_sim, delta_t, self.epsilon
        )
        cc_rate = fwd_rate[..., None] + adj
        return self.tau, cc_rate

    def df(
        self, t: float | np.float64 | str | np.datetime64 | list | np.ndarray
    ) -> np.ndarray:
        t = np.atleast_1d(self.curvenode._norm_t(t, as_of=self.as_of))
        delta_t = t - self.t_sim

        tau_x, cc_rate_y = self._knots
        idx = np.searchsorted(tau_x, t, side="right") - 1
        idx = np.clip(idx, 0, len(tau_x) - 2)
        x0 = tau_x[idx]
        x1 = tau_x[idx + 1]
        w = (t - x0) / (x1 - x0)
        y0 = cc_rate_y[idx, :]
        y1 = cc_rate_y[idx + 1, :]
        cc_rate = y0 + w[..., None] * (y1 - y0)

        df = np.exp(-cc_rate * delta_t[..., None])
        return np.where(delta_t[..., None] >= 0, df, np.nan)

    def rate(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        compounding: str | None = None,
    ) -> np.ndarray:
        t = np.atleast_1d(self.curvenode._norm_t(t, as_of=self.as_of))
        delta_t = t - self.t_sim
        return Compound.get(compounding).rate_from_df(
            self.df(t), delta_t[..., None]
        )

    def forward_df(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        T: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        **align_arr_kwargs,
    ) -> np.ndarray:
        t = np.atleast_1d(self.curvenode._norm_t(t, as_of=self.as_of))
        T = np.atleast_1d(self.curvenode._norm_t(T, as_of=self.as_of))
        t, T = align_arr(t, T, **align_arr_kwargs)
        invalid_mask = (t < self.t_sim) | (T < self.t_sim)
        t_ravel = t.ravel()
        if t_ravel.size > 1 and t_ravel[0] == t_ravel[-1] and np.all(t_ravel == t_ravel[0]):
            df_t_1 = self.df(t_ravel[:1])
            df_t = np.broadcast_to(df_t_1, (*t.shape, df_t_1.shape[-1]))
        else:
            df_t = self.df(t_ravel).reshape(*t.shape, -1)
        df_T = self.df(T.ravel()).reshape(
            *T.shape, -1
        )
        result = np.divide(
            df_T, df_t, out=np.full_like(df_T, np.nan), where=df_t != 0
        )
        return np.where(invalid_mask[..., None], np.nan, result)

    def forward_rate(
        self,
        t: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        T: float | np.float64 | str | np.datetime64 | list | np.ndarray,
        compounding: str | None = None,
        dt: float | np.float64 = 1e-6,
        **align_arr_kwargs,
    ) -> np.ndarray:
        t = np.atleast_1d(self.curvenode._norm_t(t, as_of=self.as_of))
        T = np.atleast_1d(self.curvenode._norm_t(T, as_of=self.as_of))
        t, T = align_arr(t, T, **align_arr_kwargs)
        T = np.where(np.isclose(t, T), T + dt, T)
        fwd_df = self.forward_df(t, T)
        return Compound.get(compounding).rate_from_df(
            fwd_df, (T - t)[..., None]
        )

    @classmethod
    def get(
        cls,
        curve_name: str,
        as_of: str | np.datetime64,
        sim: str | np.datetime64,
        epsilon: float | np.float64 | list | np.ndarray | None = None,
        lookback: int = 10,
        selected_col: list[str] | slice | None = None,
    ) -> "SimZeroCurve | ZeroCurve":
        curve_name = curve_name.lower()
        as_of_d = np.datetime64(as_of).astype("datetime64[D]")
        sim_d = np.datetime64(sim).astype("datetime64[D]")
        if sim_d <= as_of_d:
            return ZeroCurve.get(
                curve_name,
                sim_d,
                lookback=lookback,
                selected_col=selected_col,
            )
        cache_key = (curve_name, as_of_d, sim_d)
        if cache_key in cls.SIMZEROCURVE_CACHE:
            cached = cls.SIMZEROCURVE_CACHE[cache_key]
            if epsilon is not None:
                cached.epsilon = epsilon
            return cached
        zerocurve = ZeroCurve.get(
            curve_name,
            as_of_d,
            lookback=lookback,
            selected_col=selected_col,
        )
        model = HullWhite.get(curve_name)
        szc = cls(
            zerocurve=zerocurve,
            sim=sim_d,
            model=model,
            epsilon=epsilon,
        )
        cls.SIMZEROCURVE_CACHE[cache_key] = szc
        return szc
