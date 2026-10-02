import numpy as np
import pandas as pd
from quantmr.utils.helpers import (
    align_arr,
    norm_str,
    get_year,
    is_leap_year,
    day_of_year,
)


class DayCount:
    """
    Year fraction calculator for a day count convention.

    Supported conventions: act360, act365 and actactisda (matched after norm_str normalisation).
    Instances are cached per convention; use DayCount.get.

    Attributes:
        CONVENTIONS (set[str]): Supported normalised convention names.
        DAYCOUNT_CACHE (dict[str, DayCount]): Instances cached by normalised convention name.
    """
    CONVENTIONS = {"act360", "act365", "actactisda"}
    DAYCOUNT_CACHE: dict[str, "DayCount"] = {}

    def __init__(self, convention: str | None = None):
        """
        Store the convention name; it is validated when first used.

        Args:
            convention (str | None): Day count convention; None means act365.
        """
        self._convention = convention

    @property
    def convention(self) -> str:
        """
        Normalised convention name.

        Returns:
            str: One of CONVENTIONS; act365 if none was given.

        Raises:
            ValueError: If the convention is not supported.
        """
        convention = norm_str(self._convention or "act365")
        if convention not in self.CONVENTIONS:
            raise ValueError(
                f"Unsupported day count convention: {self._convention}"
            )
        return convention

    @staticmethod
    def _ndays(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Number of calendar days from start to end."""
        return (end - start) / np.timedelta64(1, "D")

    @staticmethod
    def _act_360(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Actual/360 year fraction."""
        return DayCount._ndays(start, end) / 360.0

    @staticmethod
    def _act_365(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Actual/365 (fixed) year fraction."""
        return DayCount._ndays(start, end) / 365.0

    @staticmethod
    def _act_act_isda(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Actual/Actual ISDA year fraction: days in each calendar year over that year's length."""
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
        """
        Year fraction between start and end dates under the convention.

        Inputs are cast to datetime64[D] and broadcast together with align_arr. Negative results
        are floored at 0, and pairs where end is before start return NaN.

        Args:
            start (str | np.datetime64 | list | np.ndarray): Start date(s).
            end (str | np.datetime64 | list | np.ndarray): End date(s).
            **align_arr_kwargs: Passed on to align_arr.

        Returns:
            np.ndarray: Year fractions as float64.

        Raises:
            ValueError: If start and end cannot be broadcast together, or the convention is not
                supported.
        """
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
        """
        Return the cached DayCount for a convention, creating it on first use.

        Args:
            convention (str | None): Day count convention; None means act365.

        Returns:
            DayCount: The shared instance for that convention.
        """
        key = norm_str(convention or "act365")
        if key not in cls.DAYCOUNT_CACHE:
            cls.DAYCOUNT_CACHE[key] = cls(convention)
        return cls.DAYCOUNT_CACHE[key]



def tenor_to_yearfrac(
        tenor: str,
        convention: str,
        start_date: np.datetime64 | pd.Timestamp,
):
    """
    Convert a tenor to a year fraction from a start date.

    The tenor is rolled from start_date with pd.DateOffset (no business day adjustment) and the
    year fraction is taken under the given convention.

    Args:
        tenor (str): Tenor label: nY, nM, nW, or one ending in 'N' (ON, one day).
        convention (str): Day count convention (see DayCount).
        start_date (np.datetime64 | pd.Timestamp): Start date.

    Returns:
        np.ndarray: One-element array with the year fraction.

    Raises:
        ValueError: If the tenor is not supported.
    """
    maturity_dates=[]
    if tenor.endswith("Y"):
        n = int(tenor[:-1])
        maturity_dates.append(start_date + pd.DateOffset(years=n))
    elif tenor.endswith("M"):
        n = int(tenor[:-1])
        maturity_dates.append(start_date + pd.DateOffset(months=n))
    elif tenor.endswith("N"):
        n = 1
        maturity_dates.append(start_date + pd.DateOffset(days=n))
    elif tenor.endswith("W"):
        n = int(tenor[:-1])
        maturity_dates.append(start_date + pd.DateOffset(weeks=n))
    else:
        raise ValueError(f"Unsupported tenor: {tenor}")
    start = np.array([start_date] * len(maturity_dates), dtype="datetime64[D]")
    end = np.array(maturity_dates, dtype="datetime64[D]")

    yf = DayCount.get(convention).yearfrac(start, end)
    return yf

