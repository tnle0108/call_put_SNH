import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path.cwd().parents[0]))

from src.bond_schedule import adjust_following
from Quant_Lib.curves import BenchmarkCurve

TENOR_MONTHS = {
    "3M": 3,
    "6M": 6,
    "9M": 9,
    "1Y": 12,
    "15M": 15,
    "18M": 18,
    "21M": 21,
    "2Y": 24,
    "27M": 27,
    "30M": 30,
    "33M": 33,
    "3Y": 36,
    "5Y": 60,
    "7Y": 84,
    "10Y": 120,
    "15Y": 180,
    "20Y": 240,
    "30Y": 360,
}
tenor_order = list(TENOR_MONTHS.keys())

def get_bond_tenor(
    issue_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    min_tolerance: int = 7,
    direction: str = "lower",
):
    if maturity_date <= issue_date:
        raise ValueError(
            f"maturity_date ({maturity_date.date()}) "
            f"must be after issue_date ({issue_date.date()})"
        )

    if direction not in {"lower", "upper"}:
        raise ValueError(
            "direction must be either 'lower' or 'upper'"
        )

    tenor_dates = [
        (
            tenor,
            issue_date + pd.DateOffset(months=months),
            months,
        )
        for tenor, months in TENOR_MONTHS.items()
    ]
    nearest = min(tenor_dates, key=lambda x: abs((x[1] - maturity_date).days))

    nearest_tenor = nearest[0]
    nearest_date = nearest[1]

    tolerance = abs((nearest_date - maturity_date).days)

    if tolerance <= min_tolerance:
        return nearest_tenor
    if direction == "lower":
        candidates = [x for x in tenor_dates if x[1] <= maturity_date]
        if not candidates:
            raise ValueError(
                f"Maturity date {maturity_date.date()} "
                f"nhỏ hơn tenor nhỏ nhất."
            )

        return max(candidates, key=lambda x: x[1])[0]

    else: 
        candidates = [x for x in tenor_dates if x[1] >= maturity_date]
        if not candidates:
            raise ValueError(
                f"Maturity date {maturity_date.date()} "
                f"lớn hơn tenor lớn nhất."
            )

        return min(candidates, key=lambda x: x[1])[0]

@dataclass
class BufferYTM:
    bond: pd.Series
    min_date: pd.Timestamp
    vbma_bond_fi: pd.DataFrame
    holiday_calendar: pd.DataFrame

    def calc_ytm_df(self):
        fixed_rate = [float(x) for x in str(self.bond['annual_coupon_rate']).split(";")]
        coupon_change_date = pd.to_datetime(str(self.bond['coupon_change_date']).split(";"), format='mixed')
        issue_date = max(
            pd.to_datetime(self.bond["issue_date"]),
            pd.to_datetime(self.min_date)
        )
        maturity_date = adjust_following(pd.to_datetime(self.bond['maturity_date']), holidays=self.holiday_calendar)

        bond_tenor = get_bond_tenor(issue_date=issue_date, maturity_date=maturity_date)

        if len(coupon_change_date) > 0 and coupon_change_date.notna().any():
            change_tenor = [get_bond_tenor(issue_date, d) for d in coupon_change_date if pd.notna(d)]
        else:
            change_tenor = []

        buffer_tenor = [bond_tenor] + change_tenor
        buffer_tenor = sorted(buffer_tenor, key=lambda x: TENOR_MONTHS[x])
    
        if len(buffer_tenor) != len(fixed_rate):
            raise ValueError(
                f"len(buffer_tenor)={len(buffer_tenor)} "
                f"khác len(fixed_rate)={len(fixed_rate)}"
            )

        tenor_rate = sorted(zip(buffer_tenor, fixed_rate), key=lambda x: TENOR_MONTHS[x[0]])
        print(tenor_rate)
        benchmark = self.vbma_bond_fi.copy()
        result = benchmark.copy()

        benchmark.index = pd.to_datetime(benchmark.index)
        result.index = benchmark.index

        available_dates = benchmark.index[
            benchmark.index <= issue_date
        ]

        if len(available_dates) == 0:
            raise ValueError(
                f"No benchmark data available on or before "
                f"issue_date {issue_date}"
            )

        reset_date = available_dates[-1]
        margins = {}

        for tenor, rate in tenor_rate:

            if tenor not in benchmark.columns:
                raise ValueError(
                    f"Benchmark does not contain tenor {tenor}"
                )

            margins[tenor] = (
                rate - benchmark.loc[reset_date, tenor]
            )

        print("reset_date =", reset_date)
        print("margins =", margins)
        for j, (tenor, _) in enumerate(tenor_rate):

            if j + 1 < len(tenor_rate):

                next_tenor = tenor_rate[j + 1][0]
                next_month = TENOR_MONTHS[next_tenor]

                columns = [
                    col
                    for col in TENOR_MONTHS
                    if TENOR_MONTHS[col] < next_month
                    and col in result.columns
                ]

            else:
                current_month = TENOR_MONTHS[tenor]

                columns = [
                    col
                    for col in TENOR_MONTHS
                    if TENOR_MONTHS[col] >= current_month
                    and col in result.columns
                ]

            margin = margins[tenor]

            result.loc[:, columns] = (
                result.loc[:, columns] + margin
            )

        return result

    def calc_zyc_df(self):
        ytm_df = self.calc_ytm_df()
        benchmark_curve = BenchmarkCurve(
            curve_name="FI ZYC VND",
            benchmark_price=ytm_df,
        )
        zyc_df_list = []
        for rpd in ytm_df.index:
            zyc_df_long = benchmark_curve.print_curve(rpd)
            zyc_df_wide = (
                zyc_df_long
                .loc[tenor_order, "value"]
                .rename(rpd)
            )
            zyc_df_list.append(zyc_df_wide)
        zyc_df = pd.DataFrame(zyc_df_list)
        zyc_df.index.name = "Date"
        return zyc_df
