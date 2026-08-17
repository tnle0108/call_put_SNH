import pandas as pd
import numpy as np
from typing import Optional, Callable, Union
from Quant_Lib.instrus.base_instrument import BaseInstrument
from Quant_Lib.instrus.currency_pair import CurrencyPair
from Quant_Lib.helpers.date_utils.paymentdate import paymentdate
from Quant_Lib.configs import FX_SWAPS


class FXSwapInstrument(BaseInstrument):
    """Foreign Exchange Swap Instrument for pricing and valuation.
    
    This class represents an FX swap instrument that extends BaseInstrument.
    It handles the calculation of near and far leg pricing, spot/forward dates,
    and provides discount factor functions for curve building.
    
    An FX swap consists of:
    - Near leg: Exchange currencies at spot rate on near date
    - Far leg: Exchange currencies back at forward rate on far date
    
    The instrument automatically initializes a currency pair dependency
    and calculates various dates (spot, start, end) based on the configuration.
    
    Attributes:
        _currency_pair_obj (CurrencyPair): Associated currency pair object
        _currency_pair_name (str): Name of the currency pair (e.g., 'USDVND')
        
    Example:
        >>> fx_swap = FXSwapInstrument('USDVND_1M')
        >>> price_series = pd.Series([23500], index=[pd.Timestamp('2023-01-15')])
        >>> df = fx_swap.df(price_series)
        >>> print(df[['near', 'far', 'function']])
    """
    
    def __init__(self, fx_swap_name: str):
        """Initialize FX Swap Instrument.
        
        Args:
            fx_swap_name (str): Name of the FX swap from FX_SWAPS configuration
            
        Raises:
            KeyError: If fx_swap_name is not found in FX_SWAPS config
        """
        super().__init__(fx_swap_name, FX_SWAPS[fx_swap_name])

    def _initialize_dependencies(self):
        """Initialize currency pair dependency."""
        self._currency_pair_name = f"{self._currency1}{self._currency2}"
        self._currency_pair_obj = CurrencyPair(currency_pair_name=self._currency_pair_name)

    def df(self, price: pd.Series):
        """Calculate discount factors DataFrame from price series."""
        self._validate_price_input(price)
        if not hasattr(self._currency_pair_obj, "_df"):
            raise ValueError("Currency pair's spot price data frame is not set.")

        df = pd.DataFrame(
            index=price.index,
            columns=["price", "spot_day", "start_day", "end_day", "function", "function_spread"],
        )
        df["near"] = self._currency_pair_obj._df["price"]
        df["price"] = price
        df["far"] = price / self._currency_pair_obj._point_value + df["near"]
        df["spot_day"] = self.spot(rpds=price.index)
        df["start_day"] = self.start(spots=df["spot_day"])
        df["end_day"] = self.end(starts=df["start_day"])
        df["function"] = df.apply(
            lambda row: self.func(
                near=row["near"],
                far=row["far"],
                end_day=row["end_day"],
            ),
            axis=1,
        )
        df["function_spread"] = df.apply(
            lambda row: self.func_spread(
                near=row["near"],
                far=row["far"],
                end_day=row["end_day"],
            ),
            axis=1,
        )
        self._df = df
        return df

    def spot(self, rpds: Union[pd.DatetimeIndex, np.ndarray]) -> pd.Series:
        """Calculate spot dates from report dates (works with datetime64)."""
        # Convert to numpy array if DatetimeIndex
        rpd_array = rpds.values if isinstance(rpds, pd.DatetimeIndex) else rpds

        match (self._ignore_pay_cal, self._settle_calendar):
            case ("Start", "VAS Accounting VN"):
                if self._spot_days == 0:
                    return pd.Series(rpd_array, index=rpds, dtype="datetime64[ns]")
                else:
                    # Returns np.ndarray of datetime64[D]
                    spots = paymentdate(
                        st=rpd_array,
                        tenor_code=f"{self._spot_days}D",
                        date_method="None",
                        curr=self._settle_calendar,
                        calendar=None,
                    )
                    # Convert to datetime64[ns] for consistency
                    return pd.Series(
                        spots.astype("datetime64[ns]"), index=rpds, dtype="datetime64[ns]"
                    )
            case _:
                raise NotImplementedError("This combination is not implemented yet")

    def start(self, spots: pd.Series) -> pd.Series:
        """Calculate start dates from spot dates (works with datetime64)."""
        match (self._ignore_pay_cal, self._settle_calendar):
            case ("Start", "VAS Accounting VN"):
                # Returns np.ndarray of datetime64[D]
                starts = paymentdate(
                    st=spots.values,
                    tenor_code=self._start.upper(),
                    date_method="None",
                    curr=self._settle_calendar,
                    calendar=None,
                )
                # Convert to datetime64[ns] for consistency
                return pd.Series(
                    starts.astype("datetime64[ns]"), index=spots.index, dtype="datetime64[ns]"
                )
            case _:
                raise NotImplementedError("This combination is not implemented yet")

    def end(self, starts: pd.Series) -> pd.Series:
        """Calculate end dates from start dates (works with datetime64)."""
        match (self._ignore_pay_cal, self._settle_calendar):
            case ("Start", "VAS Accounting VN"):
                # Returns np.ndarray of datetime64[D]
                ends = paymentdate(
                    st=starts.values,
                    tenor_code=self._end.upper(),
                    date_method=self._pay_method,
                    curr=f"{self._currency1}{self._currency2}",
                    spec_feat=self._roll_conv,
                    calendar=None,
                )
                # Convert to datetime64[ns] for consistency
                return pd.Series(
                    ends.astype("datetime64[ns]"), index=starts.index, dtype="datetime64[ns]"
                )
            case _:
                raise NotImplementedError("This combination is not implemented yet")

    def func(
        self,
        near: float,
        far: float,
        end_day: Union[np.datetime64, pd.Timestamp],
    ) -> Callable:
        """Generate pricing function for curve building."""

        def func(
            get_discount_factor: Callable,
            near=near,
            far=far,
            end_day=end_day,
            get_dependency_curve_discount_factor: Optional[Callable] = None,
        ):
            if get_dependency_curve_discount_factor is None:
                return None

            discount_factor = get_discount_factor(end_day)
            dependency_discount_factor = get_dependency_curve_discount_factor(end_day)
            if self._currency1 == "USD":
                return near / far - discount_factor / dependency_discount_factor
            elif self._currency2 == "USD":
                return far / near - discount_factor / dependency_discount_factor
            else:
                raise NotImplementedError("A currency pair without USD is not supported yet")

        return func

    def func_spread(
        self, near: float, far: float, end_day: Union[np.datetime64, pd.Timestamp]
    ) -> Callable:
        """Generate spread pricing function for curve building."""

        def func(
            get_discount_factor: Callable,
            near=near,
            far=far,
            end_day=end_day,
        ):
            discount_factor = get_discount_factor(end_day)
            if self._currency1 == "USD":
                result = near / far - discount_factor
            elif self._currency2 == "USD":
                result = far / near - discount_factor
            else:
                raise NotImplementedError("A currency pair without USD is not supported yet")

            return result

        return func
