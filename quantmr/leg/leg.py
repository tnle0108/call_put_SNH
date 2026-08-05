from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np
from quantmr.schedule import Schedule
from quantmr.curve import ZeroCurve, SimZeroCurve
from quantmr.utils import DayCount, Compound, norm_str


class Leg(ABC):

    def __init__(
        self,
        start: str | np.datetime64 | None = None,
        end: str | np.datetime64 | None = None,
        term: str | None = None,
        freq: str | None = None,
        roll: int | str | None = None,
        calendar: str | None = None,
        busadjust: str | None = None,
        eom: bool = False,
        notional: float | np.float64 | None = None,
        currency: str | None = None,
        convention: str | None = None,
        compounding: str | None = None,
        notional_exchange: bool = False,
        disc_curve: str | None = None,
        disc_is_sim: bool = False,
        disc_curve_as_of: str | np.datetime64 | None = None,
        **kwargs,
    ):
        self.schedule = Schedule(
            start=start,
            end=end,
            term=term,
            freq=freq,
            roll=roll,
            calendar=calendar,
            busadjust=busadjust,
            eom=eom,
        )
        self._notional = notional
        self._currency = currency
        self._convention = convention
        self._compounding = compounding
        self._notional_exchange = notional_exchange
        self._disc_curve = disc_curve
        self._disc_curve_is_sim = disc_is_sim
        self._disc_curve_as_of = disc_curve_as_of
        self.kwargs = kwargs

    @property
    def notional(self) -> np.float64:
        return np.float64(self._notional if self._notional is not None else 1.0)

    @property
    def currency(self) -> str:
        return norm_str(self._currency or "usd")

    @property
    def daycount(self) -> DayCount:
        return DayCount.get(self._convention)

    @property
    def compound(self) -> Compound:
        return Compound.get(self._compounding)

    def _disc_curve_at(
        self, date: np.datetime64, epsilon: np.ndarray | None = None
    ) -> ZeroCurve | SimZeroCurve:
        if self._disc_curve is None:
            raise ValueError("disc_curve must be set before discounting")
        if self._disc_curve_is_sim:
            if self._disc_curve_as_of is None:
                raise ValueError(
                    "disc_curve_as_of must be set for simulation curves"
                )
            return SimZeroCurve.get(
                self._disc_curve, self._disc_curve_as_of, date,
                epsilon=epsilon,
            )
        return ZeroCurve.get(self._disc_curve, self._disc_curve_as_of or date)

    @cached_property
    def period_yearfrac(self) -> np.ndarray:
        return self.daycount.yearfrac(
            self.schedule.astarts, self.schedule.aends
        )

    def period_discfact(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        disc_epsilon: np.ndarray | None = None,
    ) -> np.ndarray:
        vd = np.atleast_1d(
            np.asarray(value_date, dtype="datetime64[D]")
        ).ravel()
        aends = self.schedule.aends
        slices: list[np.ndarray] = []
        for d in vd:
            s = self._disc_curve_at(d, epsilon=disc_epsilon).forward_df(
                d, aends
            )
            slices.append(s)
        return np.stack(slices, axis=0)

    def _past_mask(
        self, value_date: str | np.datetime64 | list | np.ndarray
    ) -> np.ndarray:
        vd = np.atleast_1d(
            np.asarray(value_date, dtype="datetime64[D]")
        ).ravel()
        return self.schedule.aends[None, :] < vd[..., None]

    def _add_final_notional(self, cf: np.ndarray) -> np.ndarray:
        if not self._notional_exchange:
            return cf
        M = len(self.schedule.aends)
        exchange = np.zeros(M, dtype=np.float64)
        exchange[-1] = self.notional
        if cf.ndim <= 1:
            return cf + exchange
        shape = [1] * cf.ndim
        shape[1] = M
        return cf + exchange.reshape(shape)

    @abstractmethod
    def cashflow(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        ...

    def pv(
        self,
        value_date: str | np.datetime64 | list | np.ndarray,
        disc_epsilon: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        cf = self.cashflow(value_date, **kwargs)
        disc = self.period_discfact(
            value_date, disc_epsilon=disc_epsilon
        )
        if disc.ndim > cf.ndim:
            cf = cf[..., None]
        elif cf.ndim > disc.ndim:
            disc = disc[..., None]
        return np.nansum(cf * disc, axis=1)
