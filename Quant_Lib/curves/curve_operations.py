"""Curve calculation and query operations."""

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union
from Quant_Lib.helpers.date_utils.numdays import numdays
from Quant_Lib.helpers.date_utils.paymentdate import paymentdate


class CurveOperationsMixin:
    """Mixin providing calculation and query operations for curves."""

    def get_zero_rate_using_end_date(
        self, 
        rpd: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray], 
        end_date: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get zero rate for a specific end date.
        
        Parameters:
        -----------
        rpd : scalar or array-like
            Reference pricing date(s)
        end_date : scalar or array-like
            End date(s)
            
        Returns:
        --------
        float or np.ndarray
            Zero rate(s) for the given date(s)
        """
        # Check if inputs are arrays
        rpd_is_array = isinstance(rpd, (pd.DatetimeIndex, np.ndarray, list))
        end_date_is_array = isinstance(end_date, (pd.DatetimeIndex, np.ndarray, list))
        
        # Scalar case (backward compatible)
        if not rpd_is_array and not end_date_is_array:
            rpd_key = pd.Timestamp(rpd) if isinstance(rpd, np.datetime64) else rpd
            yf = numdays(st=rpd, ed=end_date, conv=self._day_count)
            return self._curve_factors[rpd_key](yf)
        
        # Convert to arrays if needed
        if not rpd_is_array:
            rpd = np.array([rpd])
            single_rpd = True
        else:
            rpd = np.asarray(rpd)
            single_rpd = False
            
        if not end_date_is_array:
            end_date = np.array([end_date])
            single_end_date = True
        else:
            end_date = np.asarray(end_date)
            single_end_date = False
        
        # Broadcast arrays if needed
        if single_rpd and not single_end_date:
            rpd = np.repeat(rpd, len(end_date))
        elif not single_rpd and single_end_date:
            end_date = np.repeat(end_date, len(rpd))
        elif len(rpd) != len(end_date):
            raise ValueError(f"rpd and end_date must have the same length or one must be scalar. Got {len(rpd)} and {len(end_date)}")
        
        # Vectorized calculation
        yfs = numdays(st=rpd, ed=end_date, conv=self._day_count)
        
        # Apply curve factors for each date
        results = np.zeros(len(rpd))
        for i in range(len(rpd)):
            rpd_key = pd.Timestamp(rpd[i]) if isinstance(rpd[i], np.datetime64) else rpd[i]
            results[i] = self._curve_factors[rpd_key](yfs[i])
        
        return results

    def get_discount_factor_using_end_date(
        self, 
        rpd: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray], 
        end_date: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Get discount factor for a specific end date.
        
        Parameters:
        -----------
        rpd : scalar or array-like
            Reference pricing date(s)
        end_date : scalar or array-like
            End date(s)
            
        Returns:
        --------
        float or np.ndarray
            Discount factor(s) for the given date(s)
        """
        # Check if inputs are arrays
        rpd_is_array = isinstance(rpd, (pd.DatetimeIndex, np.ndarray, list))
        end_date_is_array = isinstance(end_date, (pd.DatetimeIndex, np.ndarray, list))
        
        # Scalar case (backward compatible)
        if not rpd_is_array and not end_date_is_array:
            rpd_key = pd.Timestamp(rpd) if isinstance(rpd, np.datetime64) else rpd
            yf = numdays(st=rpd, ed=end_date, conv=self._day_count)
            return np.exp(-self._curve_factors[rpd_key](yf) * yf)
        
        # Convert to arrays if needed
        if not rpd_is_array:
            rpd = np.array([rpd])
            single_rpd = True
        else:
            rpd = np.asarray(rpd)
            single_rpd = False
            
        if not end_date_is_array:
            end_date = np.array([end_date])
            single_end_date = True
        else:
            end_date = np.asarray(end_date)
            single_end_date = False
        
        # Broadcast arrays if needed
        if single_rpd and not single_end_date:
            rpd = np.repeat(rpd, len(end_date))
        elif not single_rpd and single_end_date:
            end_date = np.repeat(end_date, len(rpd))
        elif len(rpd) != len(end_date):
            raise ValueError(f"rpd and end_date must have the same length or one must be scalar. Got {len(rpd)} and {len(end_date)}")
        
        # Vectorized calculation
        yfs = numdays(st=rpd, ed=end_date, conv=self._day_count)
        
        # Apply curve factors for each date
        results = np.zeros(len(rpd))
        for i in range(len(rpd)):
            rpd_key = pd.Timestamp(rpd[i]) if isinstance(rpd[i], np.datetime64) else rpd[i]
            results[i] = np.exp(-self._curve_factors[rpd_key](yfs[i]) * yfs[i])
        
        return results

    def print_curve(self, rpd: Union[np.datetime64, pd.Timestamp, str]) -> pd.DataFrame:
        """Print curve values for a given report date."""
        if isinstance(rpd, str):
            rpd = pd.to_datetime(rpd)
        elif isinstance(rpd, np.datetime64):
            rpd = pd.Timestamp(rpd)

        try:
            benchmark_objects = self.benchmark_objects
        except (AttributeError, ValueError):
            benchmark_objects = self._spread_curve.benchmark_objects

        curve_df = pd.DataFrame(
            columns=["benchmark_price", "used_date", "value"], index=benchmark_objects.keys()
        )
        for benchmark_name, benchmark_object in benchmark_objects.items():
            benchmark_price = benchmark_object._df.loc[rpd, "price"]
            used_date = benchmark_object._df.loc[rpd, "pay_day"]
            value = self.get_zero_rate_using_end_date(rpd=rpd, end_date=used_date)
            curve_df.loc[benchmark_name] = [benchmark_price, used_date, value]

        return curve_df
    
    def get_curve_at_benchmark(
        self, rpd: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray]
    ) -> pd.DataFrame:
        """Get curve values at benchmark dates for a given report date."""
        rpd = process_date_input(rpd)

        try:
            benchmark_objects = self.benchmark_objects
        except (AttributeError, ValueError):
            benchmark_objects = self._spread_curve.benchmark_objects

        curve_df = pd.DataFrame(columns=benchmark_objects.keys(), index=rpd)
        for benchmark_name, benchmark_object in benchmark_objects.items():
            used_date = benchmark_object._df.loc[rpd, "pay_day"]
            curve_df.loc[:, benchmark_name] = self.get_zero_rate_using_end_date(
                rpd=rpd, end_date=used_date
            )

        return curve_df
            

    def get_timeseries(
        self,
        rpd: Union[np.datetime64, pd.Timestamp, pd.DatetimeIndex, np.ndarray],
        tenors: list[str],
    ) -> pd.DataFrame:
        """Generate time series of zero rates for specified tenors."""
        if not hasattr(self, "_curve_factors") or self._curve_factors is None:
            raise ValueError("Curve factors have not been calculated yet.")

        # Convert to DatetimeIndex for column names
        if isinstance(rpd, (np.datetime64, pd.Timestamp)):
            columns = [pd.Timestamp(rpd)]
        elif isinstance(rpd, np.ndarray):
            columns = pd.DatetimeIndex([pd.Timestamp(d) for d in rpd])
        else:
            columns = rpd

        timeseries_df = pd.DataFrame(index=tenors, columns=columns)

        # Vectorized processing for each tenor across all dates
        for tenor in tqdm(tenors, desc=f"Calculating timeseries for {self._curve_name}"):
            rpd_dates = timeseries_df.columns

            # Vectorized paymentdate calculation for all dates at once
            end_dates = paymentdate(
                st=rpd_dates,
                tenor_code=tenor,
                date_method="None",
                curr="VAS Accounting VN",
                calendar=None,
            )

            # Vectorized numdays calculation for all dates at once
            yfs = numdays(st=rpd_dates, ed=end_dates, conv=self._day_count)

            # Apply curve factors for each date
            for i, rpd_i in enumerate(rpd_dates):
                timeseries_df.loc[tenor, rpd_i] = self._curve_factors[rpd_i](yfs[i])

        return timeseries_df
