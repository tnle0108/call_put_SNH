import numpy as np
from quantmr.utils.helpers import (
    align_arr,
    norm_str,
    get_year,
    is_leap_year,
    day_of_year,
)


class DayCount:
    CONVENTIONS = {"act360", "act365", "actactisda"}
    DAYCOUNT_CACHE: dict[str, "DayCount"] = {}

    def __init__(self, convention: str | None = None):
        self._convention = convention

    @property
    def convention(self) -> str:
        convention = norm_str(self._convention or "act365")
        if convention not in self.CONVENTIONS:
            raise ValueError(
                f"Unsupported day count convention: {self._convention}"
            )
        return convention

    @staticmethod
    def _ndays(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return (end - start) / np.timedelta64(1, "D")

    @staticmethod
    def _act_360(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return DayCount._ndays(start, end) / 360.0

    @staticmethod
    def _act_365(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return DayCount._ndays(start, end) / 365.0

    @staticmethod
    def _act_act_isda(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        start_years = get_year(start)
        end_years = get_year(end)
        start_is_leap = is_leap_year(start)
        end_is_leap = is_leap_year(end)
        diys = np.where(start_is_leap, 366.0, 365.0)
        diye = np.where(end_is_leap, 366.0, 365.0)
        start_doy = day_of_year(start)
        end_doy = day_of_year(end)
        return (
            end_years
            - start_years
            - 1
            + (diys - start_doy + 1) / diys
            + (end_doy - 1) / diye
        )

    def yearfrac(
        self,
        start: str | np.datetime64 | list | np.ndarray,
        end: str | np.datetime64 | list | np.ndarray,
        **align_arr_kwargs,
    ) -> np.ndarray:
        start_arr = np.asarray(start, dtype="datetime64[D]")
        end_arr = np.asarray(end, dtype="datetime64[D]")
        try:
            start_arr, end_arr = align_arr(
                start_arr, end_arr, **align_arr_kwargs
            )
        except ValueError as e:
            raise ValueError(
                f"Start and end dates could not be broadcast together: {e}"
            ) from e
        match self.convention:
            case "act360":
                yf = self._act_360(start_arr, end_arr)
            case "act365":
                yf = self._act_365(start_arr, end_arr)
            case "actactisda":
                yf = self._act_act_isda(start_arr, end_arr)
            case _:
                raise ValueError(
                    f"Unsupported day count convention: {self.convention}"
                )
        neg_mask = end_arr < start_arr
        yf = np.maximum(yf, 0.0)
        return np.where(neg_mask, np.nan, yf).astype(np.float64)

    @classmethod
    def get(cls, convention: str | None = None) -> "DayCount":
        key = norm_str(convention or "act365")
        if key not in cls.DAYCOUNT_CACHE:
            cls.DAYCOUNT_CACHE[key] = cls(convention)
        return cls.DAYCOUNT_CACHE[key]
