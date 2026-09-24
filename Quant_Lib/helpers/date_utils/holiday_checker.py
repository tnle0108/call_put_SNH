import pandas as pd
import numpy as np
import os
from functools import lru_cache
from datetime import date
from typing import Union


class HolidayChecker:
    """Efficient holiday checking with optimized caching and O(1) lookups."""

    # Class-level cache
    _holidays = None
    _vn_weekend_wkd = None
    _hld_log = None
    _holiday_sets = {}  # O(1) lookup sets by year and currency
    _vn_weekend_wkd_sets = {}  # O(1) lookup sets for VN working weekends
    _metadata_dir = None  # Cached metadata directory path

    @classmethod
    def _get_metadata_dir(cls):
        """Get metadata directory path (cached)."""
        if cls._metadata_dir is None:
            # Always resolve relative to the Quant_Lib root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            quant_lib_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
            cls._metadata_dir = os.path.join(quant_lib_root, "metadata", "HLD")
        return cls._metadata_dir

    @classmethod
    def _load_log(cls):
        """Load holiday log from xlsx file if not already loaded."""
        if cls._hld_log is None:
            hld_log_path = os.path.join(cls._get_metadata_dir(), "HLD_update_log.xlsx")
            cls._hld_log = pd.read_excel(hld_log_path, index_col=0, header=0)
            cls._hld_log["Start"] = pd.to_datetime(cls._hld_log["Start"])
            cls._hld_log["End"] = pd.to_datetime(cls._hld_log["End"])
            print("HLD log loaded")

    @classmethod
    def _load_holidays(cls):
        """Load holiday dates from xlsx files if not already loaded."""
        if cls._holidays is None:
            cls._load_log()
            metadata_dir = cls._get_metadata_dir()

            cls._holidays = {}
            for year in cls._hld_log.index:
                file_path = os.path.join(metadata_dir, cls._hld_log.at[year, "HLD_file"])

                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Holiday file not found: {file_path}")

                # Read and convert dates
                df = pd.read_excel(file_path, header=0)
                cls._holidays[year] = df.apply(pd.to_datetime)

                # Create O(1) lookup sets
                cls._holiday_sets[year] = {
                    col: set(d.date() for d in pd.to_datetime(df[col]).dropna())
                    for col in df.columns
                }
            print("Holidays loaded")

    @classmethod
    def _load_vn_weekend_wkd(cls):
        """Load VN weekend working days from xlsx files if not already loaded."""
        if cls._vn_weekend_wkd is None:
            cls._load_log()
            metadata_dir = cls._get_metadata_dir()

            cls._vn_weekend_wkd = {}
            for year in cls._hld_log.index:
                file_path = os.path.join(
                    metadata_dir, cls._hld_log.at[year, "VN_working_weekend_file"]
                )

                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"VN weekend working days file not found: {file_path}")

                # Read and create sets
                df = pd.read_excel(file_path, header=0)
                cls._vn_weekend_wkd[year] = df.apply(pd.to_datetime)

                # Create O(1) lookup set
                cls._vn_weekend_wkd_sets[year] = {
                    d.date() for col in df.columns for d in pd.to_datetime(df[col]).dropna()
                }
            print("VN weekend working days loaded")

    @classmethod
    @lru_cache(maxsize=2048)
    def is_holiday(
        cls,
        date: Union[date, pd.Timestamp],
        curr: str = "VAS Accounting VN",
        calendar: Union[int, None] = None,
    ) -> bool:
        """Check if the given date is a holiday (cached for performance).

        Uses O(1) set-based lookup with LRU cache for optimal performance.

        Args:
            date: Date to check (datetime, pd.Timestamp, or date object)
            curr: Currency code (default: "VAS Accounting VN")
            calendar: Calendar identifier (defaults to most recent)

        Returns:
            bool: True if date is a holiday, False otherwise
        """
        cls._load_holidays()
        cls._load_vn_weekend_wkd()

        # Normalize currency code
        if curr == "United States":
            curr = "USD"

        # Convert to date object for caching (hashable)
        if hasattr(date, "date"):
            date = date.date()

        # Get calendar and sets
        if calendar is None:
            calendar = max(cls._holidays.keys())

        holiday_sets = cls._holiday_sets[calendar]
        vn_weekend_wkd_set = cls._vn_weekend_wkd_sets[calendar]

        # Check if weekend (0=Monday, 6=Sunday)
        is_weekend = date.weekday() in [5, 6]

        # Currency-specific logic
        if curr == "VND":
            is_usdvnd_holiday = date in holiday_sets.get("USDVND", set())
            return (is_usdvnd_holiday or is_weekend) and (date not in vn_weekend_wkd_set)
        elif curr == "USD":
            is_usd_holiday = date in holiday_sets.get("USD", set())
            return is_usd_holiday or is_weekend
        elif "USD" not in curr and "VND" in curr:
            foreign_curr = (
                f"USD{curr[:3]}" if f"USD{curr[:3]}" in holiday_sets else f"{curr[:3]}USD"
            )
            is_foreign_holiday = date in holiday_sets.get(foreign_curr, set())
            is_usdvnd_holiday = date in holiday_sets.get("USDVND", set())
            return is_foreign_holiday or is_usdvnd_holiday or is_weekend
        else:
            is_curr_holiday = date in holiday_sets.get(curr, set())
            is_usd_holiday = date in holiday_sets.get("USD", set())
            return is_curr_holiday or is_usd_holiday or is_weekend

    @classmethod
    @lru_cache(maxsize=2048)
    def is_vn_weekend_working_day(
        cls, dates: Union[date, pd.Timestamp, pd.Series], calendar: Union[int, None] = None
    ) -> Union[bool, pd.Series]:
        """Check if the given date is a VN weekend working day (cached for performance).

        Args:
            dates: Date to check (single date or Series)
            calendar: Calendar identifier (defaults to most recent)

        Returns:
            bool or Series: True if date is a VN weekend working day
        """
        cls._load_vn_weekend_wkd()

        # Convert to date object for caching
        if hasattr(dates, "date"):
            dates = dates.date()

        if calendar is None:
            calendar = max(cls._vn_weekend_wkd.keys())

        vn_weekend_wkd_set = cls._vn_weekend_wkd_sets[calendar]

        # Handle single date vs Series
        if isinstance(dates, pd.Series):
            return dates.isin(vn_weekend_wkd_set)
        else:
            return dates in vn_weekend_wkd_set


if __name__ == "__main__":
    from holiday_checker import HolidayChecker
    import pandas as pd

    try:
        test_date = pd.to_datetime("2024-04-30").date()
        print(HolidayChecker.is_holiday(test_date, "VND"))
        print(HolidayChecker.is_holiday(pd.to_datetime("2024-02-08").date(), "VND"))

    except FileNotFoundError as e:
        print(e)
