from functools import cached_property
import numpy as np
import pandas as pd
from pandas import DataFrame
from quantmr.config import load_data, load_spec
from quantmr.curve.curvenode import find_date_with_lookback
from quantmr.utils import DayCount


class FX:

    FX_CACHE: dict[tuple, "FX"] = {}
    FX_DATA: dict[str, DataFrame] = {}

    def __init__(
        self,
        as_of: str | np.datetime64 | None = None,
        spot: float | np.float64 | None = None,
        convention: str | None = None,
        **kwargs,
    ):
        self._as_of = as_of
        self._spot = spot
        self._convention = convention
        self.kwargs = kwargs

    @property
    def as_of(self) -> np.datetime64:
        return np.datetime64(self._as_of or "NaT").astype("datetime64[D]")

    @property
    def spot(self) -> np.float64:
        return np.float64(self._spot if self._spot is not None else np.nan)

    @cached_property
    def daycount(self) -> DayCount:
        return DayCount.get(self._convention)

    @classmethod
    def _load_fx_data(cls) -> DataFrame:
        if "fx" not in cls.FX_DATA:
            loaded = load_data("fx").get("fx")
            cls.FX_DATA["fx"] = (
                loaded.copy() if isinstance(loaded, DataFrame) else DataFrame()
            )
        return cls.FX_DATA["fx"]

    @classmethod
    def get(
        cls,
        fx_key: tuple[str, str | np.datetime64],
        lookback: int = 10,
    ) -> "FX":
        pair_name, as_of_raw = fx_key
        pair_name = pair_name.lower()
        col_name = pair_name.upper()
        as_of = np.datetime64(as_of_raw).astype("datetime64[D]")
        cache_key = (pair_name, as_of)
        if cache_key in cls.FX_CACHE:
            return cls.FX_CACHE[cache_key]
        fx_data = cls._load_fx_data()
        if col_name not in fx_data.columns:
            raise ValueError(
                f"FX pair '{pair_name}' not found. "
                f"Available: {[c.lower() for c in fx_data.columns]}"
            )
        avail_dates = fx_data.index.values.astype("datetime64[D]")
        found_date = find_date_with_lookback(as_of, avail_dates, lookback)
        if found_date is None:
            raise ValueError(
                f"FX as_of date {as_of} not found for {pair_name} "
                f"(lookback={lookback} days)"
            )
        spot = np.float64(float(fx_data.loc[pd.Timestamp(found_date), col_name]))
        fx_spec = load_spec("fx").get(pair_name, {})
        fx_obj = cls(as_of=as_of, spot=spot, **fx_spec)
        cls.FX_CACHE[cache_key] = fx_obj
        return fx_obj


class SimFX:

    SIMFX_CACHE: dict[tuple, "SimFX"] = {}

    def __init__(
        self,
        fx: FX,
        sim: str | np.datetime64 | None = None,
        mu: float | np.float64 | None = None,
        sigma: float | np.float64 | None = None,
        epsilon: float | np.float64 | list | np.ndarray | None = None,
        **kwargs,
    ):
        self.fx = fx
        self._sim = sim
        self._mu = mu
        self._sigma = sigma
        self._epsilon = epsilon
        self.kwargs = kwargs

    @property
    def as_of(self) -> np.datetime64:
        return self.fx.as_of

    @property
    def sim(self) -> np.datetime64:
        sim = np.datetime64(self._sim or self.as_of).astype("datetime64[D]")
        if sim < self.as_of:
            raise ValueError("Sim date cannot be before as_of date")
        return sim

    @property
    def t_sim(self) -> np.float64:
        return np.float64(self.fx.daycount.yearfrac(self.as_of, self.sim))

    @property
    def mu(self) -> np.float64:
        return np.float64(self._mu if self._mu is not None else np.nan)

    @property
    def sigma(self) -> np.float64:
        sigma = np.float64(self._sigma if self._sigma is not None else np.nan)
        if sigma < 0.0:
            raise ValueError("Sigma must be non-negative")
        return sigma

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
        for attr in ("ln_spot", "spot"):
            if attr in self.__dict__:
                del self.__dict__[attr]

    @cached_property
    def _mean_lns(self) -> np.float64:
        return self.mu * self.t_sim

    @cached_property
    def _var_lns(self) -> np.float64:
        return self.sigma**2 * self.t_sim

    @cached_property
    def ln_spot(self) -> np.ndarray:
        ln_s0 = np.log(self.fx.spot)
        return ln_s0 + self._mean_lns + np.sqrt(self._var_lns) * self.epsilon

    @cached_property
    def spot(self) -> np.ndarray:
        return np.exp(self.ln_spot)

    @classmethod
    def get(
        cls,
        simfx_key: tuple[str, str | np.datetime64, str | np.datetime64],
        epsilon: float | np.float64 | list | np.ndarray | None = None,
        lookback: int = 10,
    ) -> "SimFX | FX":
        pair_name, as_of_raw, sim_raw = simfx_key
        pair_name = pair_name.lower()
        as_of = np.datetime64(as_of_raw).astype("datetime64[D]")
        sim = np.datetime64(sim_raw).astype("datetime64[D]")
        if sim <= as_of:
            return FX.get((pair_name, sim), lookback=lookback)
        cache_key = (pair_name, as_of, sim)
        if cache_key in cls.SIMFX_CACHE:
            cached = cls.SIMFX_CACHE[cache_key]
            if epsilon is not None:
                cached.epsilon = epsilon
            return cached
        fx = FX.get((pair_name, as_of), lookback=lookback)
        fx_spec = load_spec("fx").get(pair_name, {})
        sfx = cls(
            fx=fx,
            sim=sim,
            epsilon=epsilon,
            **fx_spec,
        )
        cls.SIMFX_CACHE[cache_key] = sfx
        return sfx
