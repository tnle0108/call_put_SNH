import pandas as pd
import numpy as np
from typing import Callable, Union, Optional
from Quant_Lib.instrus.base_instrument import BaseInstrument
from Quant_Lib.helpers.date_utils.paymentdate import paymentdate
from Quant_Lib.configs import CURR_SWAPS


class CurrSwap(BaseInstrument):
    def __init__(self, curr_swap_name: str):
        super().__init__(curr_swap_name, CURR_SWAPS[curr_swap_name])

    def df(self, price: pd.Series):
        """Calculate discount factors DataFrame from price series."""
        self._validate_price_input(price)
        df = pd.DataFrame(
            index=price.index,
            columns=["price", "function"],
        )

        df["price"] = price
        df["end_day"] = self.end_day(rpds=price.index)
        df["function"] = df.apply(
            lambda row: self.func(
                rpd=row.name,
                price=row["price"],
            ),
            axis=1,
        )

        self._df = df
        return df

    def end_day(self, rpds: Union[pd.DatetimeIndex, np.ndarray]) -> pd.Series:
        """Quick development for quick use, need to make it more general later."""
        # Convert to numpy array if DatetimeIndex
        rpd_array = rpds.values if isinstance(rpds, pd.DatetimeIndex) else rpds

        # Returns np.ndarray of datetime64[D]
        end_day = paymentdate(
            st=rpd_array,
            tenor_code=self._end.upper(),
            date_method="None",
            curr="VAS Accounting VN",
            calendar=None,
        )
        # Convert to datetime64[ns] for consistency
        return pd.Series(end_day.astype("datetime64[ns]"), index=rpds, dtype="datetime64[ns]")

    def func(
        self,
        rpd: Union[np.datetime64, pd.Timestamp],
        price: float,
    ) -> Callable:
        """Quick development for quick use, need to make it more general later."""

        def func(
            get_discount_factor: Callable,
            get_dependency_curve_discount_factor: Optional[Callable] = None,
        ):
            result = -1
            last_pay_day = rpd
            for i in range(int(self._end[:-1]) * 4):
                pay_day = paymentdate(
                    st=rpd,
                    tenor_code=f"{(i+1)*3}M",
                    date_method="None",
                    curr="VAS Accounting VN",
                    calendar=None,
                )

                # Convert timedelta64 to days (int)
                days_diff = int((pay_day - last_pay_day) / np.timedelta64(1, "D"))

                result += get_discount_factor(end_day=pay_day) * price * days_diff / 365

                last_pay_day = pay_day

                if i == int(self._end[:-1]) * 4 - 1:
                    result += get_discount_factor(end_day=pay_day)
            return result

        return func
