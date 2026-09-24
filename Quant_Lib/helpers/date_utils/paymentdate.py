"""Payment date calculation with business day adjustments.

Automatic single/batch detection, LRU caching, vectorized operations.
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from collections import Counter
from .holiday_checker import HolidayChecker as hc
from .date_adjustments import (
    _ensure_initialized,
    get_magnitude_delta_type,
    next_wkd_if_today_not_wkd,
    prev_wkd_if_today_not_wkd,
    preserve_eom,
    eom_cal_days,
    DateCalculator,
)

# ============================================================================
# Configuration
# ============================================================================

VALID_DATE_METHODS = frozenset(["None", "Following", "Mod. Following"])
VALID_SPEC_FEATURES = frozenset(["None", "Preserve EOM", "EOM Cal Days"])


# ============================================================================
# Core Payment Date Calculation
# ============================================================================


def _process_eom_feature(start_array, spec_feat, tenor_code, curr, calendar):
    """Process EOM special features for numpy array of dates.

    Returns: (eom_dates array, triggered_flags array)
    """
    eom_func = (
        preserve_eom
        if spec_feat == "Preserve EOM"
        else eom_cal_days if spec_feat == "EOM Cal Days" else None
    )

    if not eom_func:
        return np.full(len(start_array), np.datetime64("NaT"), dtype="datetime64[D]"), np.zeros(
            len(start_array), dtype=bool
        )

    # Apply EOM function to each date and convert results to datetime64[D]
    results = [eom_func(pd.Timestamp(d).date(), tenor_code, curr, calendar) for d in start_array]

    eom_dates = np.empty(len(results), dtype="datetime64[D]")
    triggered = np.empty(len(results), dtype=bool)

    for i, (date_result, is_triggered) in enumerate(results):
        if date_result is not None:
            eom_dates[i] = np.datetime64(date_result, "D")
        else:
            eom_dates[i] = np.datetime64("NaT")
        triggered[i] = is_triggered

    return eom_dates, triggered


def _calculate_base_dates_vectorized(
    start_array: np.ndarray, magnitude: int, delta_type: str
) -> np.ndarray:
    """Calculate base dates using numpy operations.

    Args:
        start_array: numpy array of datetime64[D]
        magnitude: int
        delta_type: str ('D', 'W', 'M', 'Y')

    Returns:
        numpy array of datetime64[D]
    """
    if delta_type == "D":
        return start_array + np.timedelta64(magnitude, "D")
    elif delta_type == "W":
        return start_array + np.timedelta64(magnitude * 7, "D")
    elif delta_type == "M":
        # Use pandas DateOffset for proper month arithmetic that preserves day
        result = pd.to_datetime(start_array) + pd.DateOffset(months=magnitude)
        return result.values.astype("datetime64[D]")
    elif delta_type == "Y":
        # Use pandas DateOffset for proper year arithmetic that preserves day
        result = pd.to_datetime(start_array) + pd.DateOffset(years=magnitude)
        return result.values.astype("datetime64[D]")
    else:
        raise ValueError(f"Delta type {delta_type} is not supported.")


def _apply_date_method_vectorized(base_dates, date_method, delta_type, curr, calendar):
    """Apply date adjustment method to numpy array of dates.

    Args:
        base_dates: numpy array of datetime64[D]
        date_method: str
        delta_type: str
        curr: str
        calendar: int or str

    Returns:
        numpy array of datetime64[D]
    """
    if date_method == "None":
        return base_dates

    # Apply following adjustment - convert Timestamps to datetime64[D]
    result = np.empty(len(base_dates), dtype="datetime64[D]")
    for i, d in enumerate(base_dates):
        adjusted = next_wkd_if_today_not_wkd(pd.Timestamp(d).date(), curr, calendar)
        result[i] = np.datetime64(adjusted.date(), "D")

    # For Mod. Following, check if month changed and use preceding instead
    if date_method == "Mod. Following" and delta_type in ["M", "Y"]:
        result_months = np.array([pd.Timestamp(d).month for d in result])
        base_months = np.array([pd.Timestamp(d).month for d in base_dates])
        month_changed = result_months != base_months

        if month_changed.any():
            for i in np.where(month_changed)[0]:
                adjusted = prev_wkd_if_today_not_wkd(
                    pd.Timestamp(base_dates[i]).date(), curr, calendar
                )
                result[i] = np.datetime64(adjusted.date(), "D")

    return result


def _should_apply_vn_compensatory_adjustment_for_this_curr(curr: str) -> bool:
    """Determine if currency type qualifies for VN compensatory weekend adjustment.
    
    Only applies to currency PAIRS (e.g., USDVND, USDJPY), not single currencies.
    Single currencies like "VAS Accounting VN", "United States", or 3-letter codes skip adjustment.
    """
    # Skip for single currencies
    if curr in ["VAS Accounting VN", "United States"] or len(curr) == 3:
        return False
    # Apply for currency pairs (6+ characters, e.g., USDVND)
    return True


def _is_exception_date_for_vn_compensatory(curr: str, date) -> bool:
    """Check if a specific date is an exception for VN compensatory adjustment.
    
    Returns True if this date should skip VN compensatory adjustment despite being a VN weekend working day.
    Special exception: USDVND on 2026-01-10 skips adjustment (FX transactions occurred that day).
    """
    if curr == "USDVND":
        # Normalize date to date object for comparison
        date_to_check = date.date() if hasattr(date, "date") else date
        exception_date = pd.Timestamp("2026-01-10").date()
        return date_to_check == exception_date
    return False


def _adjust_for_vn_compensatory_weekend(
    start_date, tenor_code, date_method, curr, spec_feat, calendar
):
    """Handle VN compensatory weekend adjustment for currencies with foreign factors to Vietnam.

    Returns:
        numpy datetime64[D]
    """
    # Check if this currency and date should get VN compensatory adjustment
    if not _should_apply_vn_compensatory_adjustment_for_this_curr(curr) or \
       _is_exception_date_for_vn_compensatory(curr, start_date) or \
       not hc.is_vn_weekend_working_day(start_date, calendar=calendar):
        result = _compute_single_paymentdate(
            start_date, tenor_code, date_method, curr, spec_feat, calendar
        )
        return np.datetime64(result, "D")
    
    # Get preceding VN working day (go back one day then find previous working day)
    prev_day_np = np.datetime64(start_date, "D") - np.timedelta64(1, "D")
    prev_day = prev_day_np.astype(object)
    preceding_vn_wkd = prev_wkd_if_today_not_wkd(prev_day, curr="VND", calendar=calendar)

    compensatory_diff_days = (
        np.datetime64(start_date, "D") - np.datetime64(preceding_vn_wkd, "D")
    ).astype("timedelta64[D]")

    result = _compute_single_paymentdate(
        (
            preceding_vn_wkd
            if not hasattr(preceding_vn_wkd, "date")
            else preceding_vn_wkd.date() if hasattr(preceding_vn_wkd, "date") else preceding_vn_wkd
        ),
        tenor_code,
        date_method,
        curr,
        spec_feat,
        calendar,
    )

    result_np = np.datetime64(result, "D") + compensatory_diff_days
    return result_np


@lru_cache(maxsize=2048)
def _compute_single_paymentdate(start_date, tenor_code, date_method, curr, spec_feat, calendar):
    """Internal cached computation for single payment date.

    start_date must be a date object (hashable for caching).
    """
    start_ts = pd.to_datetime(start_date)
    magnitude, delta_type = get_magnitude_delta_type(tenor_code)

    # Apply USD special rule
    if curr == "USD" and tenor_code[-1] != "D":
        start_ts = next_wkd_if_today_not_wkd(ori_date=start_date, curr="USD", calendar=calendar)

    # Apply EOM feature if applicable
    eom_func = (
        preserve_eom
        if spec_feat == "Preserve EOM"
        else eom_cal_days if spec_feat == "EOM Cal Days" else None
    )

    if eom_func:
        eom_date, triggered = eom_func(start_date, tenor_code, curr, calendar)
        base_date = (
            eom_date
            if triggered
            else DateCalculator.calculate_base_date(start_ts, magnitude, delta_type)
        )
    else:
        base_date = DateCalculator.calculate_base_date(start_ts, magnitude, delta_type)

    return DateCalculator.apply_date_method(base_date, date_method, delta_type, curr, calendar)


def _prepare_vn_compensatory_adjustment(start_array, curr, calendar):
    """Prepare adjustment for VN compensatory weekends (currencies with foreign factors to Vietnam).

    Args:
        start_array: numpy array of datetime64[D]
        curr: str
        calendar: int or str

    Returns:
        (effective_start_array, compensatory_adjustment) both as numpy datetime64[D] and timedelta64
    """
    compensatory_diff = np.zeros(len(start_array), dtype="timedelta64[D]")

    # Use unified logic - check if currency qualifies for adjustment
    if not _should_apply_vn_compensatory_adjustment_for_this_curr(curr):
        return start_array, compensatory_diff
    
    # Check each date for VN weekend working day AND exception dates
    compensatory_mask = np.array(
        [
            hc.is_vn_weekend_working_day(
                pd.Timestamp(d).date() if isinstance(d, np.datetime64) else d, 
                calendar=calendar
            ) and not _is_exception_date_for_vn_compensatory(curr, d)
            for d in start_array
        ],
        dtype=bool,
    )

    if compensatory_mask.any():
        # Get preceding VN working day (go back one day then find previous working day)
        preceding_vn_wkd = np.empty(compensatory_mask.sum(), dtype="datetime64[D]")
        for i, date_val in enumerate(start_array[compensatory_mask]):
            prev_day = date_val - np.timedelta64(1, "D")
            prev_day_obj = pd.Timestamp(prev_day).date()
            preceding = prev_wkd_if_today_not_wkd(prev_day_obj, curr="VND", calendar=calendar)
            preceding_vn_wkd[i] = np.datetime64(preceding, "D")

        compensatory_diff[compensatory_mask] = (
            start_array[compensatory_mask] - preceding_vn_wkd
        ).astype("timedelta64[D]")
        effective_start = start_array.copy()
        effective_start[compensatory_mask] = preceding_vn_wkd
        return effective_start, compensatory_diff

    return start_array, compensatory_diff


def paymentdate(
    st,
    tenor_code: str,
    date_method: str,
    curr: str = "VAS Accounting VN",
    spec_feat: str = "None",
    calendar=None,
) -> np.datetime64 | np.ndarray:
    """Calculate payment date with business day adjustment.

    Auto-detects single/array input. Single dates use cached version, arrays use pure numpy vectorized ops.

    For non-VND/VAS currencies, automatically adjusts for Vietnamese compensatory weekend working days
    by maintaining consistent tenor length from preceding VN working day.

    Args:
        st: Start date(s) - np.datetime64, pd.Timestamp, date, pd.Series, list, or np.ndarray
        tenor_code: Tenor code ('3M', '1Y', etc.)
        date_method: Adjustment method ('None', 'Following', 'Mod. Following')
        curr: Currency code (default: 'VAS Accounting VN')
        spec_feat: Special feature ('None', 'Preserve EOM', 'EOM Cal Days')
        calendar: Calendar identifier

    Returns:
        np.datetime64[D] for single date, np.ndarray of datetime64[D] for arrays (date only, no time)
    """
    _ensure_initialized()

    # Validate inputs
    if date_method not in VALID_DATE_METHODS:
        raise ValueError(
            f"Invalid date_method '{date_method}'. Must be one of {VALID_DATE_METHODS}"
        )
    if spec_feat not in VALID_SPEC_FEATURES:
        raise ValueError(f"Invalid spec_feat '{spec_feat}'. Must be one of {VALID_SPEC_FEATURES}")

    # Check if input is array-like
    is_array = isinstance(st, (pd.Series, pd.Index, pd.DatetimeIndex, list, np.ndarray))

    if not is_array:
        # Single date - convert to date object for computation
        if isinstance(st, np.datetime64):
            start_date = st.astype("datetime64[D]").astype(object)
        elif hasattr(st, "date"):
            start_date = st.date()
        else:
            start_date = pd.to_datetime(st).date()

        if calendar is None:
            calendar = max(hc._holidays.keys())

        result_np = _adjust_for_vn_compensatory_weekend(
            start_date, tenor_code, date_method, curr, spec_feat, calendar
        )
        return result_np

    # Array processing - convert to numpy datetime64[D] array immediately
    if isinstance(st, (list, pd.Series, pd.Index, pd.DatetimeIndex)):
        start_array = pd.to_datetime(st).values.astype("datetime64[D]")
    elif isinstance(st, np.ndarray):
        start_array = pd.to_datetime(st).values.astype("datetime64[D]")
    else:
        start_array = np.array(st, dtype="datetime64[D]")

    magnitude, delta_type = get_magnitude_delta_type(tenor_code)

    # Handle VN compensatory weekend (special case for currencies with foreign factors to Vietnam)
    effective_start_array, compensatory_adjustment = _prepare_vn_compensatory_adjustment(
        start_array, curr, calendar
    )

    # Get calendar for all dates if not specified
    if calendar is None:
        calendar = max(hc._holidays.keys())

    # Apply USD special rule if needed
    # if curr == "USD" and tenor_code[-1] != "D":
    #     effective_start_result = np.empty(len(effective_start_array), dtype="datetime64[D]")
    #     for i, d in enumerate(effective_start_array):
    #         adjusted = next_wkd_if_today_not_wkd(
    #             pd.Timestamp(d).date(), curr="USD", calendar=calendar
    #         )
    #         effective_start_result[i] = np.datetime64(adjusted.date(), "D")
    #     effective_start_array = effective_start_result

    # Process EOM special features
    eom_dates, eom_triggered = _process_eom_feature(
        effective_start_array, spec_feat, tenor_code, curr, calendar
    )

    # Calculate base dates (use EOM if triggered, otherwise normal calculation)
    base_dates_normal = _calculate_base_dates_vectorized(
        effective_start_array, magnitude, delta_type
    )

    # Combine EOM and normal base dates
    base_dates = np.where(eom_triggered, eom_dates, base_dates_normal)

    # Apply date adjustment methods
    result = _apply_date_method_vectorized(base_dates, date_method, delta_type, curr, calendar)

    # Apply VN compensatory weekend adjustment (if any)
    result = result + compensatory_adjustment

    return result


if __name__ == "__main__":
    print(
        paymentdate(
            pd.Timestamp("2025-07-31"),
            "1M",
            "Mod. Following",
            "USD",
            spec_feat="EOM Cal Days",
        )
    )

def paymentdate_no_adjustment(
    st,
    tenor_code: str,
) -> np.datetime64 | np.ndarray:
    """
    Calculate raw tenor date without any business-day adjustment.

    Unlike paymentdate(), this function:
    - does NOT adjust weekends
    - does NOT adjust holidays
    - does NOT apply Following / Mod. Following
    - does NOT apply VN compensatory weekend adjustment
    - simply adds the tenor to the start date

    Examples
    --------
    2026-03-02 + 3M  -> 2026-06-02
    2026-03-02 + 6M  -> 2026-09-02
    2026-03-02 + 1Y  -> 2027-03-02

    Returns
    -------
    np.datetime64[D] or np.ndarray
    """

    _ensure_initialized()

    magnitude, delta_type = get_magnitude_delta_type(tenor_code)

    # ============================================================
    # Single date
    # ============================================================
    is_array = isinstance(
        st,
        (pd.Series, pd.Index, pd.DatetimeIndex, list, np.ndarray)
    )

    if not is_array:

        # Convert everything to Timestamp
        start_ts = pd.Timestamp(st)

        if delta_type == "D":
            result = start_ts + pd.Timedelta(days=magnitude)

        elif delta_type == "W":
            result = start_ts + pd.Timedelta(weeks=magnitude)

        elif delta_type == "M":
            result = start_ts + pd.DateOffset(months=magnitude)

        elif delta_type == "Y":
            result = start_ts + pd.DateOffset(years=magnitude)

        else:
            raise ValueError(
                f"Delta type {delta_type} is not supported."
            )

        return np.datetime64(result, "D")

    # ============================================================
    # Array / Series / Index
    # ============================================================

    start_array = pd.to_datetime(st)

    if delta_type == "D":
        result = start_array + pd.to_timedelta(magnitude, unit="D")

    elif delta_type == "W":
        result = start_array + pd.to_timedelta(
            magnitude * 7,
            unit="D"
        )

    elif delta_type == "M":
        result = start_array + pd.DateOffset(months=magnitude)

    elif delta_type == "Y":
        result = start_array + pd.DateOffset(years=magnitude)

    else:
        raise ValueError(
            f"Delta type {delta_type} is not supported."
        )

    return result.values.astype("datetime64[D]")