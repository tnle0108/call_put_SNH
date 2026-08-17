import pandas as pd
import numpy as np
from typing import Callable, Union
from Quant_Lib.instrus.base_instrument import BaseInstrument
from Quant_Lib.helpers.date_utils.paymentdate import paymentdate
from Quant_Lib.helpers.date_utils.numdays import numdays
from Quant_Lib.configs import RATE_INDEX


class RateIndex(BaseInstrument):
    """Interest Rate Index Instrument for rate curve construction.
    
    This class represents interest rate indices (e.g., LIBOR, SOFR, VNIBOR)
    used in curve building and pricing. It calculates discount factors and
    provides functions for curve bootstrapping based on market rates.
    
    The rate index handles:
    - Spot date calculation from report dates
    - Start and end date determination based on tenor
    - Discount factor calculation for curve building
    - Business day adjustments and calendar handling
    
    The instrument supports various rate conventions including different
    settlement calendars, day count conventions, and rolling conventions.
    
    Attributes:
        All attributes are loaded from RATE_INDEX configuration including:
        - Settlement calendar and business day rules
        - Day count conventions 
        - Tenor and reset periods
        - Rate calculation methods
        
    Example:
        >>> rate_idx = RateIndex('VNIBOR_3M')
        >>> price_series = pd.Series([5.25], index=[pd.Timestamp('2023-01-15')])
        >>> df = rate_idx.df(price_series)
        >>> print(df[['price', 'spot_day', 'end_day', 'function']])
    """
    
    def __init__(self, rate_index_name: str):
        """Initialize Rate Index Instrument.
        
        Args:
            rate_index_name (str): Name of the rate index from RATE_INDEX configuration
            
        Raises:
            KeyError: If rate_index_name is not found in RATE_INDEX config
        """
        super().__init__(rate_index_name, RATE_INDEX[rate_index_name])

    def df(self, price: pd.Series):
        """Calculate discount factors DataFrame from price series."""
        self._validate_price_input(price)

        df = pd.DataFrame(
            index=price.index, columns=["price", "spot_day", "start_day", "end_day", "pay_day", "function"]
        )
        df["price"] = price
        df["spot_day"] = self.spot(rpds=price.index)
        df["start_day"] = self.start(spots=df["spot_day"])
        df["end_day"] = self.end(starts=df["start_day"])
        df["pay_day"] = self.pay(starts=df["start_day"])
        df["function"] = df.apply(
            lambda row: self.func(
                price=row["price"],
                start_day=row["start_day"],
                end_day=row["end_day"],
                pay_day = row["pay_day"]
            ),
            axis=1,
        )

        self._df = df
        return df

    def spot(self, rpds: Union[pd.DatetimeIndex, np.ndarray]) -> pd.Series:
        """Calculate spot dates from report dates (vectorized, works with datetime64)."""
        # Convert to numpy array if DatetimeIndex
        rpd_array = rpds.values if isinstance(rpds, pd.DatetimeIndex) else rpds

        match (self._start_adjustment, self._settle_calendar, self._ignore_pay_cal):
            case ("None", "VAS Accounting VN", "Start"):
                if self._spot_days == 0:
                    # No offset, return as datetime64 series
                    return pd.Series(rpd_array, index=rpds, dtype="datetime64[ns]")
                else:
                    # Use vectorized paymentdate - returns np.ndarray of datetime64[D]
                    spots = paymentdate(
                        st=rpd_array,
                        tenor_code=f"{self._spot_days}D",
                        date_method="None",
                        curr=self._settle_calendar,
                        calendar=None,
                    )
                    # Convert from datetime64[D] to datetime64[ns] for consistency
                    return pd.Series(
                        spots.astype("datetime64[ns]"), index=rpds, dtype="datetime64[ns]"
                    )
            case _:
                raise NotImplementedError("This combination is not implemented yet")

    def start(self, spots: pd.Series) -> pd.Series:
        """Calculate start dates from spot dates (vectorized, works with datetime64)."""
        match (self._start_adjustment, self._settle_calendar, self._ignore_pay_cal):
            case ("None", "VAS Accounting VN", "Start"):
                # Use vectorized paymentdate - returns np.ndarray of datetime64[D]
                starts = paymentdate(
                    st=spots.values,
                    tenor_code=self._start_period.upper(),
                    date_method="None",
                    curr=self._settle_calendar,
                    calendar=None,
                    spec_feat=self._roll_conv,
                )
                # Convert from datetime64[D] to datetime64[ns] for consistency
                return pd.Series(
                    starts.astype("datetime64[ns]"), index=spots.index, dtype="datetime64[ns]"
                )
            case _:
                raise NotImplementedError("This combination is not implemented yet")

    def end(self, starts: pd.Series) -> pd.Series:
        """Calculate end dates from start dates (vectorized, works with datetime64)."""
        # Use vectorized paymentdate - returns np.ndarray of datetime64[D]
        ends = paymentdate(
            st=starts.values,
            tenor_code=self._end_period.upper(),
            date_method=self._day_method,
            curr="VAS Accounting VN",
            calendar=None,
            spec_feat=self._roll_conv,
        )
        # Convert from datetime64[D] to datetime64[ns] for consistency
        return pd.Series(ends.astype("datetime64[ns]"), index=starts.index, dtype="datetime64[ns]")

    def pay(self, starts: pd.Series) -> pd.Series:
        """Calculate end dates from start dates (vectorized, works with datetime64)."""
        # Use vectorized paymentdate - returns np.ndarray of datetime64[D]
        ends = paymentdate(
            st=starts.values,
            tenor_code=self._end_period.upper(),
            date_method=self._day_method,
            curr=self._pay_cal,
            calendar=None,
            spec_feat=self._roll_conv,
        )
        # Convert from datetime64[D] to datetime64[ns] for consistency
        return pd.Series(ends.astype("datetime64[ns]"), index=starts.index, dtype="datetime64[ns]")

    def func(
        self,
        price: float,
        start_day: Union[np.datetime64, pd.Timestamp],
        end_day: Union[np.datetime64, pd.Timestamp],
        pay_day: Union[np.datetime64, pd.Timestamp],
    ) -> Callable:
        """Generate pricing function for curve building."""
        match (self._rate_type, self._quotation):

            # case ("annual", "Coupon"):

            #     def func(
            #         get_discount_factor: Callable,
            #         price=price,
            #         start_day=start_day,
            #         end_day=end_day,
            #         pay_day=pay_day,
            #     ):
            #         coupon = float(price)

            #         if abs(coupon) > 1:
            #             coupon /= 100.0

            #         start_day = pd.Timestamp(start_day)
            #         end_day = pd.Timestamp(end_day)
            #         pay_day = pd.Timestamp(pay_day)

            #         coupon_dates = []
            #         current_date = end_day
            #         while True:
            #             previous_date = current_date - pd.DateOffset(years=1)
            #             if previous_date <= start_day:
            #                 break
            #             coupon_dates.append(previous_date)
            #             current_date = previous_date
            #         coupon_dates.sort()

            #         coupon_pv = 0.0
            #         previous_date = start_day
            #         for coupon_date in coupon_dates:
            #             accrual = numdays(
            #                 st=previous_date,
            #                 ed=coupon_date,
            #                 conv=self._day_count,
            #             )
            #             df = get_discount_factor(coupon_date)
            #             coupon_pv += coupon * accrual * df
            #             previous_date = coupon_date

            #         final_accrual = numdays(
            #             st=previous_date,
            #             ed=pay_day,
            #             conv=self._day_count,
            #         )
            #         maturity_df = get_discount_factor(pay_day)
            #         return (coupon_pv + (1.0 + coupon * final_accrual) * maturity_df - 1.0)
            #     return func

            # case ("annual", "Coupon"):
            #     def func(
            #         get_discount_factor: Callable,
            #         price=price,
            #         start_day=start_day,
            #         end_day=end_day,
            #         pay_day=pay_day,
            #     ):
            #         coupon = float(price)
            #         if abs(coupon) > 1:
            #             coupon /= 100.0

            #         start_day = pd.Timestamp(start_day)
            #         end_day = pd.Timestamp(end_day)
            #         pay_day = pd.Timestamp(pay_day)

            #         previous_coupon_dates = []
            #         current_date = end_day
            #         while True:
            #             previous_date = current_date - pd.DateOffset(years=1)
            #             if previous_date <= start_day:
            #                 break
            #             previous_coupon_dates.append(previous_date)
            #             current_date = previous_date
            #         previous_coupon_dates.sort()
            #         coupon_pv = 0.0
            #         previous_date = start_day
            #         for coupon_end in previous_coupon_dates:
            #             accrual = numdays(
            #                 st=previous_date,
            #                 ed=coupon_end,
            #                 conv=self._day_count,
            #             )
            #             discount_factor = get_discount_factor(coupon_end)
            #             coupon_pv += (
            #                 coupon
            #                 * accrual
            #                 * discount_factor
            #             )

            #             previous_date = coupon_end

            #         final_accrual = numdays(
            #             st=previous_date,
            #             ed=pay_day,
            #             conv=self._day_count,
            #         )

            #         final_discount_factor = get_discount_factor(pay_day)

            #         final_cashflow = (
            #             1.0
            #             + coupon * final_accrual
            #         )

            #         return (
            #             coupon_pv
            #             + final_cashflow * final_discount_factor
            #             - 1.0
            #         )

            #     return func
            case ("annual", "Coupon"):

                def func(
                    get_discount_factor: Callable,
                    price=price,
                    start_day=start_day,
                    end_day=end_day,
                    pay_day=pay_day,
                ):
                    """
                    Pricing equation for annual coupon instrument.

                    Previous coupon cashflows:
                        discounted at END DATE

                    Final coupon + principal:
                        discounted at PAY DATE
                    """

                    coupon = float(price)

                    # Market data của bạn đang là decimal:
                    # 0.077 = 7.7%
                    #
                    # Nhưng nếu truyền 7.7 thì vẫn hỗ trợ.
                    if abs(coupon) > 1.0:
                        coupon /= 100.0

                    start_day = pd.Timestamp(start_day)
                    end_day = pd.Timestamp(end_day)
                    pay_day = pd.Timestamp(pay_day)

                    # ============================================================
                    # Build coupon dates backwards from maturity
                    #
                    # Ví dụ 15M:
                    #
                    # start = 4/8/2026
                    # end   = 7/8/2027
                    #
                    # previous coupon = 7/8/2026
                    #
                    # => stub = 3M
                    #
                    # Ví dụ 27M:
                    #
                    # end = 7/8/2028
                    # previous = 7/8/2027
                    # previous = 7/8/2026
                    # ============================================================

                    previous_coupon_dates = []

                    current_date = end_day

                    while True:
                        previous_date = current_date - pd.DateOffset(years=1)

                        if previous_date <= start_day:
                            break

                        previous_coupon_dates.append(previous_date)
                        current_date = previous_date

                    previous_coupon_dates.sort()

                    # ============================================================
                    # Previous coupons
                    #
                    # DF lấy tại END DATE
                    # ============================================================

                    coupon_pv = 0.0
                    previous_date = start_day

                    for coupon_end in previous_coupon_dates:

                        accrual = numdays(
                            st=previous_date,
                            ed=coupon_end,
                            conv=self._day_count,
                        )

                        discount_factor = get_discount_factor(
                            coupon_end
                        )

                        coupon_pv += (
                            coupon
                            * accrual
                            * discount_factor
                        )

                        previous_date = coupon_end

                    # ============================================================
                    # Final coupon
                    #
                    # Accrual: previous coupon date -> END DATE
                    #
                    # Nhưng DF:
                    #       PAY DATE
                    # ============================================================

                    final_accrual = numdays(
                        st=previous_date,
                        ed=pay_day,
                        conv=self._day_count,
                    )

                    final_cashflow = (
                        1.0
                        + coupon * final_accrual
                    )

                    final_discount_factor = get_discount_factor(
                        pay_day
                    )

                    return (
                        coupon_pv
                        + final_cashflow * final_discount_factor
                        - 1.0
                    )

                # ================================================================
                # Attach metadata để CurveBootstrapMixin có thể sequential
                # bootstrap mà không cần inspect source code.
                # ================================================================

                func._instrument_type = "annual_coupon"
                func._price = float(price)
                func._start_day = pd.Timestamp(start_day)
                func._end_day = pd.Timestamp(end_day)
                func._pay_day = pd.Timestamp(pay_day)

                return func
            case _:
                raise NotImplementedError("This combination is not implemented yet")
