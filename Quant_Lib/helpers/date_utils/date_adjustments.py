"""Date adjustment utilities for payment calculations.

Tenor parsing, working day adjustments, EOM features.
"""

import pandas as pd
import datetime as dt
from functools import lru_cache
from dateutil.relativedelta import relativedelta
from datetime import date
from typing import Tuple, Union, Optional
from .holiday_checker import HolidayChecker as hc

# ============================================================================
# Initialization
# ============================================================================

_initialized = False


def _ensure_initialized():
    """Ensure holiday data is loaded (lazy initialization)."""
    global _initialized
    if not _initialized:
        hc._load_holidays()
        hc._load_vn_weekend_wkd()
        hc._load_log()
        _initialized = True


# ============================================================================
# Tenor Parsing
# ============================================================================


@lru_cache(maxsize=256)
def get_magnitude_delta_type(tenor_code: str) -> Tuple[int, str]:
    """Convert tenor code to (magnitude, delta_type). e.g., '3M' -> (3, 'M')."""
    tenor_code = tenor_code.upper()
    if tenor_code == "ON":
        return (1, "D")
    elif tenor_code == "SW":
        return (1, "W")
    else:
        delta_type = tenor_code[-1]
        magnitude = int(tenor_code[:-1])
        return (magnitude, delta_type)


# ============================================================================
# Working Day Adjustment Functions
# ============================================================================


@lru_cache(maxsize=2048)
def next_wkd_if_today_not_wkd(
    ori_date: Union[date, pd.Timestamp], curr: str, calendar: Optional[int] = None
) -> pd.Timestamp:
    """Find next working day if current date is not a working day (cached)."""
    _ensure_initialized()

    # Convert to date object for caching
    if hasattr(ori_date, "date"):
        ori_date = ori_date.date()

    # Efficient loop with O(1) holiday checks
    while hc.is_holiday(ori_date, curr, calendar=calendar):
        ori_date = ori_date + dt.timedelta(days=1)

    return pd.Timestamp(ori_date)


@lru_cache(maxsize=2048)
def prev_wkd_if_today_not_wkd(
    ori_date: Union[date, pd.Timestamp], curr: str, calendar: Optional[int] = None
) -> pd.Timestamp:
    """Find previous working day if current date is not a working day (cached)."""
    _ensure_initialized()

    # Convert to date object for caching
    if hasattr(ori_date, "date"):
        ori_date = ori_date.date()

    while hc.is_holiday(ori_date, curr, calendar=calendar):
        ori_date = ori_date - dt.timedelta(days=1)

    return pd.Timestamp(ori_date)


# ============================================================================
# Special Features (EOM Handling)
# ============================================================================


def _is_end_of_month_working(
    date: Union[date, pd.Timestamp], curr: str, calendar: Optional[int]
) -> bool:
    """Check if date is end of month on working calendar (last working day of month)."""
    next_day = date + dt.timedelta(days=1)
    return next_wkd_if_today_not_wkd(next_day, curr, calendar=calendar).month != date.month


def _is_end_of_month_calendar(date: Union[date, pd.Timestamp]) -> bool:
    """Check if date is end of month on calendar days (last calendar day of month)."""
    next_day = date + dt.timedelta(days=1)
    return next_day.month != date.month


def _calculate_eom_date(ori_date, months):
    """Calculate end-of-month date after adding months."""
    # Move to first day of next month, then add months and subtract 1 day
    next_month = ori_date + relativedelta(months=1)
    target_month = pd.Timestamp(next_month.year, next_month.month, 1)
    return target_month + relativedelta(months=months, days=-1)


@lru_cache(maxsize=1024)
def preserve_eom(
    ori_date: Union[date, pd.Timestamp], tenor_code: str, curr: str, calendar: Optional[int] = None
) -> Tuple[Optional[date], bool]:
    """Preserve end-of-month: if ori_date is EOM, adjusted date will be too (cached).

    Uses working calendar to determine EOM (last working day of month).
    Returns: (result_date or None, is_triggered boolean)
    """
    _ensure_initialized()

    if hasattr(ori_date, "date"):
        ori_date = ori_date.date()

    magnitude, delta_type = get_magnitude_delta_type(tenor_code)

    # Only applies to M/Y tenors and EOM dates (working calendar)
    if delta_type not in ["M", "Y"] or not _is_end_of_month_working(ori_date, curr, calendar):
        return (None, False)

    if hc.is_holiday(ori_date, curr, calendar=calendar):
        return (None, False)

    months = magnitude if delta_type == "M" else magnitude * 12
    result_date = _calculate_eom_date(ori_date, months)
    return (result_date, True)


@lru_cache(maxsize=1024)
def eom_cal_days(
    ori_date: Union[date, pd.Timestamp], tenor_code: str, curr: str, calendar: Optional[int] = None
) -> Tuple[Optional[date], bool]:
    """End-of-month calendar days feature (cached).

    Uses calendar days to determine EOM (last calendar day of month).
    Returns: (result_date or None, is_triggered boolean)
    """
    _ensure_initialized()

    if hasattr(ori_date, "date"):
        ori_date = ori_date.date()

    magnitude, delta_type = get_magnitude_delta_type(tenor_code)

    # Only applies to M/Y tenors and EOM dates (calendar days)
    if delta_type not in ["M", "Y"] or not _is_end_of_month_calendar(ori_date):
        return (None, False)

    if hc.is_holiday(ori_date, curr, calendar=calendar):
        return (None, False)

    months = magnitude if delta_type == "M" else magnitude * 12
    result_date = _calculate_eom_date(ori_date, months)
    return (result_date, True)


# ============================================================================
# Date Calculation Helper
# ============================================================================


class DateCalculator:
    """Helper for date calculations."""

    @staticmethod
    def calculate_base_date(
        st: Union[date, pd.Timestamp], magnitude: int, delta_type: str
    ) -> Union[date, pd.Timestamp]:
        """Calculate base date from start date and tenor."""
        if delta_type == "D":
            return st + relativedelta(days=magnitude)
        elif delta_type == "W":
            return st + relativedelta(weeks=magnitude)
        elif delta_type == "M":
            return st + relativedelta(months=magnitude)
        elif delta_type == "Y":
            return st + relativedelta(years=magnitude)
        else:
            raise ValueError(f"Delta type {delta_type} is not supported.")

    @staticmethod
    def apply_date_method(
        base_date: Union[date, pd.Timestamp],
        date_method: str,
        delta_type: str,
        curr: str,
        calendar: Optional[int],
    ) -> pd.Timestamp:
        """Apply business day adjustment method ('None', 'Following', 'Mod. Following')."""
        base_date_obj = base_date.date() if hasattr(base_date, "date") else base_date

        if date_method == "None":
            return base_date
        elif date_method == "Following":
            return next_wkd_if_today_not_wkd(base_date_obj, curr, calendar=calendar)
        elif date_method == "Mod. Following":
            temp_res = next_wkd_if_today_not_wkd(base_date_obj, curr, calendar=calendar)
            if (temp_res.month != base_date.month) and (delta_type in ["M", "Y"]):
                return prev_wkd_if_today_not_wkd(base_date_obj, curr, calendar=calendar)
            else:
                return temp_res
        else:
            raise ValueError(f"Invalid date_method: {date_method}")
