import re
from functools import cached_property
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from quantmr.config import load_data
from quantmr.utils import norm_str, is_month_end, to_month_end


class BusCalendar:
    BUSCALENDAR_CACHE: dict[str | None, "BusCalendar"] = {}
    BUSADJUSTS = {
        None,
        "following",
        "preceding",
        "modifiedfollowing",
        "modifiedpreceding",
    }
    UNITS = {"d", "w", "m", "y"}

    def __init__(self, holiday: list | np.ndarray | None = None):
        self._holiday = holiday

    @property
    def holiday(self) -> np.ndarray:
        holiday = self._holiday if self._holiday is not None else []
        return np.asarray(holiday, dtype="datetime64[D]")

    @cached_property
    def _np_buscalendar(self) -> np.busdaycalendar:
        return np.busdaycalendar(holidays=self.holiday)

    def is_busday(
        self, date: str | np.datetime64 | list | np.ndarray
    ) -> np.ndarray:
        date = np.asarray(date, dtype="datetime64[D]")
        return np.is_busday(dates=date, busdaycal=self._np_buscalendar)

    def adjust(
        self,
        date: str | np.datetime64 | list | np.ndarray,
        busadjust: str | None = None,
    ) -> np.ndarray:
        date = np.asarray(date, dtype="datetime64[D]")
        busadjust = norm_str(busadjust) if busadjust else None
        if busadjust not in self.BUSADJUSTS:
            raise ValueError(
                f"Unsupported business day adjustment: {busadjust}"
            )
        if not busadjust:
            return date
        adjust = np.busday_offset(
            dates=date,
            offsets=0,
            roll=busadjust,
            busdaycal=self._np_buscalendar,
        )
        return np.asarray(adjust, dtype="datetime64[D]")

    def _shift_by_n_unit(
        self,
        date: np.ndarray,
        n: int = 0,
        unit: str | None = None,
        busadjust: str | None = None,
        eom: bool = False,
    ) -> np.ndarray:
        unit = norm_str(unit) if unit else None
        if unit not in self.UNITS:
            raise ValueError(f"Unsupported shift unit: {unit}")
        date_shape = date.shape
        date = date.ravel()
        date_pd = pd.to_datetime(date)
        match unit:
            case "d":
                shift_pd = date_pd + DateOffset(days=n)
            case "w":
                shift_pd = date_pd + DateOffset(weeks=n)
            case "m":
                shift_pd = date_pd + DateOffset(months=n)
            case "y":
                shift_pd = date_pd + DateOffset(years=n)
        shift = np.asarray(shift_pd, dtype="datetime64[D]")
        if eom and unit in ("m", "y"):
            shift = np.where(is_month_end(date), to_month_end(shift), shift)
        if not busadjust:
            return shift.reshape(date_shape)
        if unit in ("d", "w"):
            if busadjust == "modifiedfollowing":
                busadjust = "following"
            elif busadjust == "modifiedpreceding":
                busadjust = "preceding"
        return self.adjust(shift, busadjust=busadjust).reshape(date_shape)

    @staticmethod
    def _term_to_n_unit(
        term: str | None = None,
    ) -> list:
        if not term:
            return [None]
        term = norm_str(term)
        if term == "on":
            term = "1d"
        matches = re.findall(r"(\d+)([a-z]+)", term)
        if not matches:
            raise ValueError(f"Invalid term format: {term}")
        return [(int(n), u) for n, u in matches]

    def _shift_by_term(
        self,
        date: np.ndarray,
        term: str | None = None,
        busadjust: str | None = None,
        eom: bool = False,
        direction: str = "forward",
    ) -> np.ndarray:
        term = norm_str(term) if term else None
        direction = norm_str(direction)
        match direction:
            case "forward":
                sign = 1
            case "backward":
                sign = -1
            case _:
                raise ValueError(f"Invalid shift direction: {direction}")
        tuples = self._term_to_n_unit(term)
        shift = date
        for i, item in enumerate(tuples):
            if not item:
                continue
            n, unit = item
            is_last = i == len(tuples) - 1
            curr_busadjust = busadjust if is_last else None
            shift = self._shift_by_n_unit(
                shift, n=sign * n, unit=unit, busadjust=curr_busadjust, eom=eom
            )
        return shift

    def shift(
        self,
        date: str | np.datetime64 | list | np.ndarray,
        term: str | list | np.ndarray | None = None,
        busadjust: str | None = None,
        eom: bool = False,
        direction: str = "forward",
    ) -> np.ndarray:
        date = np.asarray(date, dtype="datetime64[D]")
        date_shape = date.shape
        term = np.asarray(term, dtype=object).ravel()
        shift = np.vectorize(
            lambda term: self._shift_by_term(
                date, term, busadjust, eom, direction
            ),
            otypes=[object],
        )(term)
        shift = np.stack([s.reshape(date_shape) for s in shift], axis=-1)
        return np.squeeze(shift)

    @classmethod
    def get(cls, calendar: str | None = None) -> "BusCalendar":
        if calendar is not None:
            calendar = norm_str(calendar)
            part = [p.strip() for p in calendar.split("+") if p.strip()]
            calendar = "+".join(sorted(part))
        if calendar not in cls.BUSCALENDAR_CACHE:
            if calendar is None:
                holiday = []
            elif "+" in calendar:
                holiday = []
                for p in part:
                    holiday_p = load_data("holiday").get(p)
                    if holiday_p is None:
                        raise ValueError(f"Holiday data not found for key: {p}")
                    holiday.append(holiday_p)
                holiday = np.unique(np.concatenate(holiday))
            else:
                holiday = load_data("holiday").get(calendar)
                if holiday is None:
                    raise ValueError(
                        f"Holiday data not found for key: {calendar}"
                    )
                if not isinstance(holiday, np.ndarray):
                    raise TypeError(
                        f"Expected ndarray for holiday '{calendar}', "
                        f"got {type(holiday).__name__}"
                    )
            buscalendar = BusCalendar(holiday=holiday)
            cls.BUSCALENDAR_CACHE[calendar] = buscalendar
        return cls.BUSCALENDAR_CACHE[calendar]
