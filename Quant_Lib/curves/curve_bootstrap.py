"""Bootstrap functionality for curve construction."""

import pandas as pd
import numpy as np
import traceback
from scipy.optimize import root
from tqdm import tqdm
from typing import Union, Callable, Tuple
from Quant_Lib.helpers.date_utils.numdays import numdays
from Quant_Lib.helpers.math_utils.interpl import interpolate


class CurveBootstrapMixin:
    """Mixin providing bootstrapping functionality for curves."""

    def get_curve_factors(self):
        """Bootstrap curve factors from benchmark prices."""
        if not hasattr(self, "_benchmark_price") or self._benchmark_price is None:
            raise ValueError("Benchmark price data is not set.")

        curve_factors = {}
        previous_zero_rates = None  # Track previous solution for warm start

        for rpd in tqdm(
            self._benchmark_price.index, desc=f"Bootstrapping curve factors for {self._curve_name}"
        ):
            yfs_array, funcs, prices = self._extract_benchmark_data(rpd)
            zero_rates = self._solve_for_zero_rates(
                rpd, yfs_array, funcs, prices, previous_zero_rates
            )
            curve_factors[rpd] = self._create_curve_factor_function(yfs_array, zero_rates)
            previous_zero_rates = zero_rates  # Use this solution as next initial guess

        self._curve_factors = curve_factors
        return curve_factors

    def _extract_benchmark_data(
        self, rpd: Union[np.datetime64, pd.Timestamp]
    ) -> Tuple[np.ndarray, list, list]:
        """Extract year fractions, pricing functions, and prices for given date."""
        # Convert to pd.Timestamp for indexing once
        rpd_key = pd.Timestamp(rpd) if isinstance(rpd, np.datetime64) else rpd

        # Pre-allocate lists with approximate size for better performance
        n_benchmarks = len(self._benchmark_objects)
        used_dates = []
        yfs = []
        funcs = []
        prices = []
        ins_nos = []

        # Cache the day count convention and curve type to avoid attribute lookups
        day_count = self._day_count
        is_spread_curve = self._curve_type == "SpreadCurve"
        func_col = "function_spread" if is_spread_curve else "function"

        for benchmark_object in self._benchmark_objects.values():
            # Use .get() to avoid exception overhead for missing data
            df_row = benchmark_object._df.loc[rpd_key]
            price = df_row["price"]

            if price is None or np.isnan(price):
                continue

            used_date = df_row["pay_day"]
            func = df_row[func_col]

            if used_date not in used_dates:
                used_dates.append(used_date)
                yfs.append(numdays(st=rpd, ed=used_date, conv=day_count))
                funcs.append(func)
                prices.append(price)
                ins_nos.append(benchmark_object._ins_no)
            else:
                # Replace with benchmark having smaller instrument number
                ins_no = benchmark_object._ins_no
                idx = used_dates.index(used_date)
                if ins_no < ins_nos[idx]:
                    funcs[idx] = func
                    prices[idx] = price
                    ins_nos[idx] = ins_no

        return np.array(yfs), funcs, prices

    def _get_benchmark_function(
        self, benchmark_object, rpd: Union[np.datetime64, pd.Timestamp]
    ) -> Callable:
        """Get pricing function from benchmark object."""
        # Convert to pd.Timestamp for indexing
        rpd_key = pd.Timestamp(rpd) if isinstance(rpd, np.datetime64) else rpd

        if self._curve_type == "SpreadCurve":
            return benchmark_object._df.loc[rpd_key, "function_spread"]
        else:
            return benchmark_object._df.loc[rpd_key, "function"]

    def _solve_for_zero_rates(
        self,
        rpd: Union[np.datetime64, pd.Timestamp],
        yfs_array: np.ndarray,
        funcs: list,
        prices: list,
        previous_zero_rates: np.ndarray = None,
    ) -> np.ndarray:
        """Solve for zero rates with warm start or algebraic initial guess."""
        has_dependency = hasattr(self, "_dependency_curve")

        if has_dependency:

            def get_dependency_curve_discount_factor(
                end_day, _rpd=rpd, _dep_curve=self._dependency_curve
            ):
                return _dep_curve.get_discount_factor_using_end_date(_rpd, end_day)

        equations = self._build_equations_function(
            rpd,
            yfs_array,
            funcs,
            has_dependency,
            get_dependency_curve_discount_factor if has_dependency else None,
        )

        # Use previous solution as initial guess if available and similar size
        if previous_zero_rates is not None and len(previous_zero_rates) == len(yfs_array):
            initial_guess = previous_zero_rates.copy()
        else:
            # Create algebraic initial guess based on closed-form solutions
            initial_guess = self._get_algebraic_initial_guess(
                rpd,
                yfs_array,
                funcs,
                prices,
                has_dependency,
                get_dependency_curve_discount_factor if has_dependency else None,
            )

        try:
            # hybr (Powell's hybrid method) with default parameters - well-tested and robust
            result = root(equations, initial_guess, method="hybr")
            return result.x
        except Exception as e:
            print(traceback.format_exc())
            raise RuntimeError(f"Root finding failed: {e} for report date {rpd}")

    # # ========== Initial Guess Methods ==========
    # def _solve_for_zero_rates(
    #     self,
    #     rpd,
    #     yfs_array,
    #     funcs,
    #     prices,
    #     previous_zero_rates=None,
    # ):
    #     """
    #     Sequential bootstrap.

    #     This matches the Excel logic:

    #         DF_n =
    #             (1 - previous_coupon_PV)
    #             / final_cashflow

    #         zero_rate_n =
    #             -ln(DF_n) / yearfrac_pay

    #     Previous coupons are discounted at End Date.
    #     Final coupon + principal is discounted at Pay Date.
    #     """

    #     n = len(yfs_array)

    #     zero_rates = np.full(n, np.nan, dtype=float)

    #     # ================================================================
    #     # Discount factor function using already bootstrapped zero rates
    #     # ================================================================

    #     def get_previous_discount_factor(end_day):

    #         yf = numdays(
    #             st=rpd,
    #             ed=end_day,
    #             conv=self._day_count,
    #         )

    #         valid = ~np.isnan(zero_rates)

    #         if not np.any(valid):
    #             raise ValueError(
    #                 f"No bootstrapped zero rates available "
    #                 f"for {rpd} / {end_day}"
    #             )

    #         x = yfs_array[valid]
    #         y = zero_rates[valid]

    #         zero_rate = interpolate(
    #             x=x,
    #             y=y,
    #             x_star=yf,
    #             short_end=self._extrapol_short_end,
    #             long_end=self._extrapol_long_end,
    #         )

    #         return np.exp(
    #             -zero_rate * yf
    #         )
    #     # ================================================================
    #     # Sequential bootstrap
    #     # ================================================================

    #     for i, (yf, func, price) in enumerate(
    #         zip(yfs_array, funcs, prices)
    #     ):

    #         # ------------------------------------------------------------
    #         # Check instrument type
    #         # ------------------------------------------------------------

    #         instrument_type = getattr(
    #             func,
    #             "_instrument_type",
    #             None,
    #         )

    #         if instrument_type != "annual_coupon":

    #             raise NotImplementedError(
    #                 f"Sequential bootstrap is currently implemented "
    #                 f"for annual_coupon only. "
    #                 f"Got: {instrument_type}"
    #             )

    #         coupon = float(
    #             getattr(func, "_price", price)
    #         )

    #         if abs(coupon) > 1.0:
    #             coupon /= 100.0

    #         start_day = pd.Timestamp(
    #             func._start_day
    #         )

    #         end_day = pd.Timestamp(
    #             func._end_day
    #         )

    #         pay_day = pd.Timestamp(
    #             func._pay_day
    #         )

    #         # ============================================================
    #         # Build previous coupon dates
    #         # ============================================================

    #         previous_coupon_dates = []

    #         current_date = end_day

    #         while True:

    #             previous_date = (
    #                 current_date
    #                 - pd.DateOffset(years=1)
    #             )

    #             if previous_date <= start_day:
    #                 break

    #             previous_coupon_dates.append(
    #                 previous_date
    #             )

    #             current_date = previous_date

    #         previous_coupon_dates.sort()

    #         # ============================================================
    #         # PV of previous coupons
    #         #
    #         # IMPORTANT:
    #         #
    #         # Previous coupon -> END DATE
    #         # ============================================================

    #         previous_coupon_pv = 0.0

    #         previous_date = start_day

    #         for coupon_end in previous_coupon_dates:

    #             accrual = numdays(
    #                 st=previous_date,
    #                 ed=coupon_end,
    #                 conv=self._day_count,
    #             )

    #             df_previous = get_previous_discount_factor(
    #                 coupon_end
    #             )

    #             previous_coupon_pv += (
    #                 coupon
    #                 * accrual
    #                 * df_previous
    #             )

    #             previous_date = coupon_end

    #         # ============================================================
    #         # Final accrual
    #         # ============================================================

    #         final_accrual = numdays(
    #             st=previous_date,
    #             ed=end_day,
    #             conv=self._day_count,
    #         )

    #         final_cashflow = (
    #             1.0
    #             + coupon * final_accrual
    #         )

    #         # ============================================================
    #         # DF at PAY DATE
    #         #
    #         # This is the unknown DF we solve for.
    #         # ============================================================

    #         df_current = (
    #             1.0 - previous_coupon_pv
    #         ) / final_cashflow

    #         # ============================================================
    #         # Year fraction to PAY DATE
    #         # ============================================================

    #         pay_yf = numdays(
    #             st=rpd,
    #             ed=pay_day,
    #             conv=self._day_count,
    #         )

    #         if pay_yf <= 0:
    #             raise ValueError(
    #                 f"Invalid Pay Date {pay_day} "
    #                 f"for report date {rpd}"
    #             )

    #         if df_current <= 0:
    #             raise ValueError(
    #                 f"Negative/non-positive DF at {rpd}, "
    #                 f"instrument={i}, "
    #                 f"end_day={end_day}, "
    #                 f"pay_day={pay_day}, "
    #                 f"df={df_current}, "
    #                 f"previous_coupon_pv={previous_coupon_pv}, "
    #                 f"final_cashflow={final_cashflow}"
    #             )

    #         # ============================================================
    #         # Excel:
    #         #
    #         #   = -LN(DF) / YearFrac(PayDate)
    #         #
    #         # ============================================================

    #         zero_rates[i] = (
    #             -np.log(df_current)
    #             / pay_yf
    #         )

    #     return zero_rates


    def _get_algebraic_initial_guess(
        self, rpd, yfs_array, funcs, prices, has_dependency, get_dependency_curve_discount_factor
    ):
        """Derive initial guess from closed-form solutions of pricing equations."""
        initial_guess = np.zeros(len(yfs_array))
        get_df_flat = self._create_flat_df_function(rpd, flat_rate=0.05)

        for i, (yf, func, price) in enumerate(zip(yfs_array, funcs, prices)):
            initial_guess[i] = self._guess_for_instrument(
                yf,
                func,
                price,
                rpd,
                get_df_flat,
                has_dependency,
                get_dependency_curve_discount_factor,
            )

        return initial_guess

    def _create_flat_df_function(self, rpd, flat_rate=0.05):
        """Create discount factor function with flat rate."""

        def get_df_flat(end_day):
            yf = numdays(st=rpd, ed=end_day, conv=self._day_count)
            return np.exp(-flat_rate * yf)

        return get_df_flat

    def _guess_for_instrument(
        self,
        yf,
        func,
        price,
        rpd,
        get_df_flat,
        has_dependency,
        get_dependency_curve_discount_factor,
    ):
        """Determine initial guess for single instrument based on its type."""
        try:
            import inspect

            source = inspect.getsource(func)

            # RateIndex: Solve DF = 1/(1+price)^yf for zero_rate ≈ ln(1+price)
            if "1 + price" in source and "**" in source:
                return self._guess_rate_index(yf, func, get_df_flat)

            # FX Swap: Solve DF_spread = (near/far) * DF_base
            elif (
                "near" in source and "far" in source
            ) or "discount_factor / dependency_discount_factor" in source:
                return self._guess_fx_swap(
                    yf, func, rpd, get_df_flat, has_dependency, get_dependency_curve_discount_factor
                )

            # Swap instruments: Use swap rate (price) as initial guess
            elif "result +=" in source and "days_diff" in source:
                return np.clip(price, 0.001, 0.15)

            else:
                # Use price if reasonable, else 5%
                return price if (price is not None and 0.001 <= price <= 0.15) else 0.05

        except:
            return 0.05

    def _guess_rate_index(self, yf, func, get_df_flat):
        """Initial guess for RateIndex instrument."""
        try:
            result = func(get_df_flat)
            implied_df = 1 / (result + 1) if (result + 1) > 0 else 0.95
            if implied_df > 0 and implied_df <= 1:
                zero_rate = -np.log(implied_df) / yf
                return np.clip(zero_rate, 0.001, 0.15)
            return 0.05
        except:
            return 0.05

    def _guess_fx_swap(
        self, yf, func, rpd, get_df_flat, has_dependency, get_dependency_curve_discount_factor
    ):
        """Initial guess for FX Swap instrument."""
        try:
            if has_dependency:
                dep_df = get_dependency_curve_discount_factor(
                    rpd + pd.Timedelta(days=int(yf * 365))
                )
                result = func(
                    get_df_flat,
                    get_dependency_curve_discount_factor=get_dependency_curve_discount_factor,
                )
            else:
                dep_df = np.exp(-0.05 * yf)
                result = func(get_df_flat)

            implied_df = np.clip(dep_df * (1 - result), 0.7, 1.0)
            zero_rate = -np.log(implied_df) / yf if yf > 0 else 0.02
            return np.clip(zero_rate, -0.01, 0.10)
        except:
            return 0.02

    # ========== Equation Building Methods ==========

    def _build_equations_function(
        self,
        rpd: Union[np.datetime64, pd.Timestamp],
        yfs_array: np.ndarray,
        funcs: list,
        has_dependency: bool,
        get_dependency_curve_discount_factor: Union[Callable, None],
    ) -> Callable:
        """Build the system of equations for root finding."""

        # Cache attributes to avoid repeated lookups in inner loop
        day_count = self._day_count
        short_end = self._extrapol_short_end
        long_end = self._extrapol_long_end

        def equations(zero_rates):
            def get_discount_factor(end_day):
                yf_star = numdays(st=rpd, ed=end_day, conv=day_count)
                zero_rate_star = interpolate(
                    x=yfs_array,
                    y=zero_rates,
                    x_star=yf_star,
                    short_end=short_end,
                    long_end=long_end,
                )
                return np.exp(-zero_rate_star * yf_star)

            # Pre-allocate array for better performance
            equations_list = np.empty(len(funcs), dtype=float)

            if has_dependency:
                for i, func in enumerate(funcs):
                    equations_list[i] = func(
                        get_discount_factor=get_discount_factor,
                        get_dependency_curve_discount_factor=get_dependency_curve_discount_factor,
                    )
            else:
                for i, func in enumerate(funcs):
                    equations_list[i] = func(get_discount_factor=get_discount_factor)

            return equations_list

        return equations

    def _create_curve_factor_function(self, yfs_array: np.ndarray, zero_rates: np.ndarray):
        """Create interpolation function for curve factors."""
        # Make copies once and store as local variables for faster access
        yfs_copy = yfs_array.copy()
        zero_rates_copy = np.array(zero_rates)
        short_end = self._extrapol_short_end
        long_end = self._extrapol_long_end

        def curve_factor(t: float | np.ndarray) -> float | np.ndarray:
            """Interpolate zero rates at time t."""
            return interpolate(
                x=yfs_copy,
                y=zero_rates_copy,
                x_star=t,
                short_end=short_end,
                long_end=long_end,
            )

        return curve_factor
