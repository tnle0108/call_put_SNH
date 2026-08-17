import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Union


# Cache for year length calculations
@lru_cache(maxsize=128)
def _ndays_year_cached(year: int) -> int:
    """Cached version of days in year calculation."""
    return (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days


# Cache for common date calculations (scalar only)
@lru_cache(maxsize=1024)
def _numdays_single_cached(
    st_tuple: tuple, ed_tuple: tuple, conv: str, type_ISMA: str = "end"
) -> float:
    """Cached calculation for single date pair."""
    st = dt.date(*st_tuple)
    ed = dt.date(*ed_tuple)
    return _calculate_numdays_core(st, ed, conv, type_ISMA)


def numdays(
    st: Union[np.datetime64, pd.Timestamp, dt.date, pd.Series, np.ndarray, list],
    ed: Union[np.datetime64, pd.Timestamp, dt.date, pd.Series, np.ndarray, list],
    conv: str,
    type_ISMA: str = "end",
) -> Union[float, np.ndarray]:
    """Calculate year fraction between dates using specified day count convention.

    Native np.datetime64 support with automatic conversion. Unified calculation for scalars and arrays.
    Vectorized operations for simple conventions, cached calculations for complex ones.

    Args:
        st: Start date(s) - np.datetime64, pd.Timestamp, date, pd.Series, np.ndarray, or list
        ed: End date(s) - np.datetime64, pd.Timestamp, date, pd.Series, np.ndarray, or list
        conv: Day count convention ('Act/ActISMA', 'Act/ActISDA', 'Act/365', 'Act/360')
        type_ISMA: Type for ISMA convention ('start' or 'end')

    Returns:
        float for scalar inputs, np.ndarray for array inputs
    """
    # Check if array-like
    is_st_array = isinstance(st, (pd.Series, pd.Index, pd.DatetimeIndex, list, np.ndarray))
    is_ed_array = isinstance(ed, (pd.Series, pd.Index, pd.DatetimeIndex, list, np.ndarray))

    if is_st_array or is_ed_array:
        # Array path - convert to numpy datetime64[D]
        st_array = _to_datetime64_array(st)
        ed_array = _to_datetime64_array(ed)

        # Fast path for simple conventions (pure numpy)
        if conv in ["Act/365", "Act/360"]:
            deno = int(conv[-3:])
            days_diff = (ed_array - st_array).astype("timedelta64[D]").astype(int)
            return days_diff / deno

        # Complex conventions - use cached calculation per element
        result = np.empty(len(st_array), dtype=float)
        for i in range(len(st_array)):
            st_date = st_array[i].astype(object)
            ed_date = ed_array[i].astype(object)
            st_tuple = (st_date.year, st_date.month, st_date.day)
            ed_tuple = (ed_date.year, ed_date.month, ed_date.day)
            result[i] = _numdays_single_cached(st_tuple, ed_tuple, conv, type_ISMA)
        return result

    # Scalar path - convert to date object
    st_date = _to_date_object(st)
    ed_date = _to_date_object(ed)

    # Use cached calculation
    st_tuple = (st_date.year, st_date.month, st_date.day)
    ed_tuple = (ed_date.year, ed_date.month, ed_date.day)
    return _numdays_single_cached(st_tuple, ed_tuple, conv, type_ISMA)


def _to_datetime64_array(dates) -> np.ndarray:
    """Convert various date types to numpy datetime64[D] array."""
    if isinstance(dates, (pd.Series, pd.Index, pd.DatetimeIndex)):
        return dates.values.astype("datetime64[D]")
    elif isinstance(dates, list):
        return np.array(pd.to_datetime(dates).values, dtype="datetime64[D]")
    elif isinstance(dates, np.ndarray):
        if dates.dtype.kind == "M":  # already datetime64
            return dates.astype("datetime64[D]")
        else:
            return np.array(pd.to_datetime(dates).values, dtype="datetime64[D]")
    else:
        return dates


def _to_date_object(date_val) -> dt.date:
    """Convert various date types to date object."""
    if isinstance(date_val, np.datetime64):
        return date_val.astype("datetime64[D]").astype(object)
    elif hasattr(date_val, "date"):
        return date_val.date()
    else:
        return date_val


def _calculate_numdays_core(st: dt.date, ed: dt.date, conv: str, type_ISMA: str = "end") -> float:
    """Core year fraction calculation logic (works for both scalar and array elements).

    Args:
        st: Start date (date object)
        ed: End date (date object)
        conv: Day count convention
        type_ISMA: ISMA convention type

    Returns:
        Year fraction as float
    """
    valid_conv = ["Act/ActISMA", "Act/ActISDA", "Act/365", "Act/360"]

    # Check logical conditions
    if st > ed:
        raise Exception("End date must be larger than start date")

    if st == ed:
        return 0

    if conv not in valid_conv:
        raise Exception("Invalid convention, using Act/365 instead")

    def act_ISMA(st, ed, type_ISMA="end"):
        # Only work with annually compounding
        # Need to determine the number of full year payments and the residuals
        # Determine the number of full year payments and the residuals requires determine the last_coupon_date and the next_coupon_date
        y_st = st.year
        m_st = st.month
        d_st = st.day
        y_ed = ed.year
        m_ed = ed.month
        d_ed = ed.day

        if type_ISMA == "start":
            try:
                compare_date = dt.date(y_ed, m_st, d_st)
            except ValueError:
                compare_date = dt.date(y_ed, m_st, d_st - 1)

            if compare_date > ed:
                try:
                    last_cp_date = dt.date(y_ed - 1, m_st, d_st)
                except ValueError:
                    last_cp_date = dt.date(y_ed - 1, m_st, d_st - 1)
                nfull_pmt = y_ed - y_st - 1
            else:
                try:
                    last_cp_date = dt.date(y_ed, m_st, d_st)
                except ValueError:
                    last_cp_date = dt.date(y_ed, m_st, d_st - 1)
                nfull_pmt = y_ed - y_st

            next_cp_date = last_cp_date + relativedelta(years=1)

            residual_pmt = (ed - last_cp_date).days / (next_cp_date - last_cp_date).days

            res = residual_pmt + nfull_pmt

        elif type_ISMA == "end":
            try:
                compare_date = dt.date(y_st, m_ed, d_ed)
            except ValueError:
                compare_date = dt.date(y_st, m_ed, d_ed - 1)

            if compare_date > st:
                try:
                    next_cp_date = dt.date(y_st, m_ed, d_ed)
                except ValueError:
                    next_cp_date = dt.date(y_st, m_ed, d_ed - 1)
                nfull_pmt = y_ed - y_st
            else:
                try:
                    next_cp_date = dt.date(y_st + 1, m_ed, d_ed)
                except ValueError:
                    next_cp_date = dt.date(y_st + 1, m_ed, d_ed - 1)
                nfull_pmt = y_ed - y_st - 1

            last_cp_date = next_cp_date - relativedelta(years=1)
            residual_pmt = (next_cp_date - st).days / (next_cp_date - last_cp_date).days
            res = residual_pmt + nfull_pmt
        return res

    def act_ISDA(st, ed):
        y_st = st.year
        y_ed = ed.year

        if y_st != y_ed:
            # first year:
            yearfrac_first_year = (dt.date(y_st + 1, 1, 1) - st).days / _ndays_year_cached(y_st)

            # number of full years:
            nfull_year = y_ed - y_st - 1

            # last year:
            yearfrac_last_year = (ed - dt.date(y_ed, 1, 1)).days / _ndays_year_cached(y_ed)

            res = yearfrac_first_year + nfull_year + yearfrac_last_year

        else:
            res = (ed - st).days / _ndays_year_cached(y_st)

        return res

    def act_days(st, ed, conv):
        deno = int(conv[-3:])
        res = (ed - st).days / deno
        return res

    match conv:
        case "Act/ActISMA":
            res_mas = act_ISMA(st, ed, type_ISMA=type_ISMA)
        case "Act/ActISDA":
            res_mas = act_ISDA(st, ed)
        case "Act/365" | "Act/360":
            res_mas = act_days(st, ed, conv)

    return res_mas


if __name__ == "__main__":
    import datetime as dt

    st = dt.date(2024, 2, 2)
    ed = dt.date(2024, 3, 3)
    print(numdays(st, ed, "Act/ISDA"))
