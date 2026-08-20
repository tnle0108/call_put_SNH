from functools import cached_property
import numpy as np
from pandas import DataFrame
from quantmr.config import load_data, load_spec
from quantmr.utils import Compound, DayCount, norm_str, to_array
from quantmr.schedule import BusCalendar


def find_date_with_lookback(
    target: np.datetime64,
    available: np.ndarray,
    lookback: int,
) -> np.datetime64 | None:
    if target in available:
        return target
    if lookback <= 0 or len(available) == 0:
        return None
    lookback_start = target - np.timedelta64(lookback, "D")
    candidates = available[(available >= lookback_start) & (available < target)]
    if len(candidates) == 0:
        return None
    return candidates.max()


CURVE_DATA_CACHE: dict[str, DataFrame] = {}


def load_curve_data(curve_name: str, refresh: bool = True) -> DataFrame:
    if refresh:
        CURVE_DATA_CACHE.pop(curve_name, None)
    if curve_name not in CURVE_DATA_CACHE:
        loaded = load_data("curve").get(curve_name)
        CURVE_DATA_CACHE[curve_name] = (
            loaded.copy() if isinstance(loaded, DataFrame) else DataFrame()
        )
    return CURVE_DATA_CACHE[curve_name]


class CurveNode:

    CURVENODE_CACHE: dict[tuple, "CurveNode"] = {}

    def __init__(
        self,
        as_of=None,
        terms: list[str] | None = None,
        values: np.ndarray | list | None = None,
        convention: str | None = None,
        compounding: str | None = None,
        calendar: str | None = None,
        busadjust: str | None = None,
        eom: bool = False,
        value_type: str | None = None,
        **kwargs,
    ):
        if as_of is None:
            self._as_of = np.array([], dtype="datetime64[D]")
        else:
            arr = np.atleast_1d(np.asarray(as_of, dtype="datetime64[D]"))
            self._as_of = arr.ravel()

        self._terms = list(terms) if terms else []

        if values is None:
            self._values = np.empty(
                (len(self._as_of), len(self._terms)), dtype=np.float64
            )
        else:
            v = np.asarray(values, dtype=np.float64)
            if v.ndim == 1:
                v = v.reshape(1, -1)
            self._values = v

        self._convention = convention
        self._compounding = compounding
        self._calendar = calendar
        self._busadjust = busadjust
        self._eom = eom
        self._value_type = value_type
        self.kwargs = kwargs

    @property
    def as_of(self) -> np.ndarray:
        return self._as_of

    @property
    def n_dates(self) -> int:
        return len(self._as_of)

    @property
    def terms(self) -> list[str]:
        return self._terms

    def date_index(self, date: str | np.datetime64) -> int:
        d = np.datetime64(date).astype("datetime64[D]")
        idx = np.searchsorted(self._as_of, d)
        if idx >= len(self._as_of) or self._as_of[idx] != d:
            raise ValueError(
                f"Date {d} not found in CurveNode "
                f"(range {self._as_of[0]}..{self._as_of[-1]})"
            )
        return int(idx)

    @cached_property
    def value_type(self) -> str:
        value_type = norm_str(self._value_type or "rate")
        if value_type not in {"rate", "df"}:
            raise ValueError(f"Unsupported value_type: {self._value_type}")
        return value_type

    @cached_property
    def daycount(self) -> DayCount:
        return DayCount.get(self._convention)

    @cached_property
    def compounding(self) -> Compound:
        return Compound.get(self._compounding)

    @cached_property
    def buscalendar(self) -> BusCalendar:
        return BusCalendar.get(self._calendar)

    @cached_property
    def end_date(self) -> np.ndarray:
        if not self._terms:
            return np.array([], dtype="datetime64[D]").reshape(self.n_dates, 0)
        _, dtype = to_array(self._terms)
        if dtype == "date":
            dates = np.asarray(self._terms, dtype="datetime64[D]")
            return np.tile(dates, (self.n_dates, 1))
        elif dtype == "str":
            raw = self.buscalendar.shift(
                date=self._as_of,
                term=self._terms,
                busadjust=self._busadjust,
                eom=self._eom,
                direction="forward",
            )
            raw = np.atleast_2d(np.asarray(raw, dtype="datetime64[D]"))
            if raw.shape != (self.n_dates, len(self._terms)):
                raw = raw.reshape(self.n_dates, -1)
            return raw
        else:
            raise ValueError(
                f"Node terms must be dates or tenor strings, got: {dtype}"
            )

    @cached_property
    def _node_value(self) -> np.ndarray:
        return self._values

    @cached_property
    def node_tau(self) -> np.ndarray:
        return self.daycount.yearfrac(self._as_of[:, None], self.end_date)

    @cached_property
    def node_df(self) -> np.ndarray:
        match self.value_type:
            case "rate":
                return self.compounding.df_from_rate(
                    self._node_value, self.node_tau
                )
            case "df":
                return self._node_value
            case _:
                raise ValueError(f"Unsupported value_type: {self.value_type}")

    @cached_property
    def node_rate(self) -> np.ndarray:
        match self.value_type:
            case "rate":
                return self._node_value
            case "df":
                return self.compounding.rate_from_df(
                    self._node_value, self.node_tau
                )
            case _:
                raise ValueError(f"Unsupported value_type: {self.value_type}")

    def node_tau_at(self, idx: int) -> np.ndarray:
        return self.node_tau[idx]

    def node_df_at(self, idx: int) -> np.ndarray:
        return self.node_df[idx]

    def node_rate_at(self, idx: int) -> np.ndarray:
        return self.node_rate[idx]

    def _norm_t(
        self,
        t,
        as_of: np.datetime64 | None = None,
    ) -> np.ndarray:
        t, dtype = to_array(t)
        match dtype:
            case "float":
                return t
            case "date":
                if as_of is None:
                    raise ValueError(
                        "as_of required to convert dates to year fractions"
                    )
                return self.daycount.yearfrac(as_of, t)
            case _:
                raise TypeError(f"Cannot convert {dtype} to year fraction")

    @classmethod
    def get(
        cls,
        curve_name: str,
        as_of_range: (
            tuple[str | np.datetime64, str | np.datetime64] | None
        ) = None,
        selected_col: list[str] | slice | None = None,
        refresh: bool = False,
    ) -> "CurveNode":
        curve_name = curve_name.lower()

        if as_of_range is not None:
            norm_range = (
                np.datetime64(as_of_range[0]).astype("datetime64[D]"),
                np.datetime64(as_of_range[1]).astype("datetime64[D]"),
            )
        else:
            norm_range = None
        if selected_col is not None:
            if isinstance(selected_col, slice):
                norm_col = (
                    selected_col.start,
                    selected_col.stop,
                    selected_col.step,
                )
            else:
                norm_col = tuple(selected_col)
        else:
            norm_col = None

        cache_key = (curve_name, norm_range, norm_col)
        if refresh:
            cls.CURVENODE_CACHE.pop(cache_key, None)
        if cache_key in cls.CURVENODE_CACHE:
            return cls.CURVENODE_CACHE[cache_key]

        curve_data = load_curve_data(curve_name)
        curve_spec = load_spec("curve").get(curve_name, {})

        if as_of_range is not None and norm_range is not None:
            start, end = norm_range
            idx = curve_data.index.values.astype("datetime64[D]")
            mask = (idx >= start) & (idx <= end)
            curve_data = curve_data.loc[mask]

        if selected_col is not None:
            if isinstance(selected_col, slice):
                curve_data = curve_data.iloc[:, selected_col]
            else:
                curve_data = curve_data[selected_col]

        as_of_dates = curve_data.index.values.astype("datetime64[D]")
        terms = list(curve_data.columns)
        values = curve_data.values.astype(np.float64)

        cn = cls(as_of_dates, terms, values, **curve_spec)
        cls.CURVENODE_CACHE[cache_key] = cn
        return cn
