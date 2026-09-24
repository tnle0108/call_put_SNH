from functools import cached_property
import numpy as np
import pandas as pd
from quantmr.schedule.buscalendar import BusCalendar
from quantmr.utils.helpers import norm_str, to_roll


class Schedule:
    FREQS_MAP: dict[str | None, str | None] = {
        None: None,
        "d": "1d",
        "w": "1w",
        "m": "1m",
        "q": "3m",
        "s": "6m",
        "y": "1y",
    }

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
    ):
        self._start = start
        self._end = end
        self._term = term
        self._freq = freq
        self._roll = roll
        self._calendar = calendar
        self._busadjust = busadjust
        self._eom = eom

    @cached_property
    def buscalendar(self) -> BusCalendar:
        return BusCalendar.get(self._calendar)

    @cached_property
    def start(self) -> np.datetime64:
        if self._start:
            return np.datetime64(self._start).astype("datetime64[D]")
        if not self._end or not self._term:
            raise ValueError("Both end and term should be provided!")
        start = self.buscalendar.shift(
            date=self._end,
            term=self._term,
            busadjust=self._busadjust,
            eom=self._eom,
            direction="backward",
        )
        return np.datetime64(start.item()).astype("datetime64[D]")

    @cached_property
    def end(self) -> np.datetime64:
        if self._end:
            return np.datetime64(self._end).astype("datetime64[D]")
        if not self._start or not self._term:
            raise ValueError("Both start and term should be provided!")
        end = self.buscalendar.shift(
            date=self._start,
            term=self._term,
            busadjust=self._busadjust,
            eom=self._eom,
            direction="forward",
        )
        return np.datetime64(end.item()).astype("datetime64[D]")

    @property
    def roll(self) -> int:
        if not self._roll:
            roll = pd.Timestamp(self.end).day
        elif isinstance(self._roll, str):
            roll = norm_str(self._roll)
            if roll == "eom":
                roll = 31
            elif roll == "som":
                roll = 1
            else:
                raise ValueError(f"Invalid roll: {roll}")
        else:
            roll = self._roll
        return roll

    @property
    def freq(self) -> str | None:
        freq = norm_str(self._freq) if self._freq else None
        if freq not in self.FREQS_MAP:
            raise ValueError(f"Unsupported frequency: {self._freq}")
        return freq

    def _generate_from_roll(
        self,
        roll_start: np.datetime64,
        roll_end: np.datetime64,
    ) -> np.ndarray:
        freq = self.FREQS_MAP[self.freq]
        if freq is None:
            return np.asarray([roll_start, roll_end], dtype="datetime64[D]")
        n = int(freq[:-1])
        unit = freq[-1]
        busadj = self._busadjust if unit in ("d", "w") else None
        dates = [roll_end]
        i = 1
        while True:
            shift = np.datetime64(
                self.buscalendar.shift(
                    date=roll_end,
                    term=f"{n * i}{unit}",
                    busadjust=busadj,
                    eom=self._eom,
                    direction="backward",
                ).item(),
            ).astype("datetime64[D]")
            if shift < roll_start:
                break
            dates.append(shift)
            i += 1
        if roll_start not in dates:
            dates.append(roll_start)
        if unit not in ("d", "w"):
            dates[:-1] = to_roll(
                np.asarray(dates[:-1], dtype="datetime64[D]"), roll=self.roll
            )
        return np.asarray(dates, dtype="datetime64[D]")

    @cached_property
    def uschedule(self) -> np.ndarray:
        freq_term = self.FREQS_MAP.get(self.freq)
        freq_unit = freq_term[-1] if freq_term else None
        if self._roll and freq_unit not in ("d", "w"):
            roll_days = to_roll(
                np.asarray([self.start, self.end], dtype="datetime64[D]"),
                roll=self.roll,
            )
            roll_start = roll_days[0]
            roll_end = roll_days[1]
        else:
            roll_start = self.start
            roll_end = self.end
        dates = self._generate_from_roll(roll_start, roll_end)
        mask = (dates >= self.start) & (dates <= self.end)
        filtered = dates[mask]
        result = filtered
        if self.start not in result:
            result = np.insert(result, 0, self.start)
        if self.end not in result:
            result = np.append(result, self.end)
        result = np.unique(result)
        return result

    @cached_property
    def aschedule(self) -> np.ndarray:
        return np.asarray(
            self.buscalendar.adjust(
                self.uschedule,
                busadjust=self._busadjust,
            ),
            dtype="datetime64[D]",
        )

    @cached_property
    def ustarts(self) -> np.ndarray:
        return self.uschedule[:-1]

    @cached_property
    def uends(self) -> np.ndarray:
        return self.uschedule[1:]

    @cached_property
    def astarts(self) -> np.ndarray:
        return self.aschedule[:-1]

    @cached_property
    def aends(self) -> np.ndarray:
        return self.aschedule[1:]

    @cached_property
    def table(self) -> pd.DataFrame:
        data = {
            "Unadjusted Start": self.ustarts,
            "Unadjusted End": self.uends,
            "Adjusted Start": self.astarts,
            "Adjusted End": self.aends,
        }
        return pd.DataFrame(data)
