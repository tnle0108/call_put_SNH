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

VBMA_TENOR_MONTHS = {
    "1Y": 12,
    "2Y": 24,
    "3Y": 36,
    "5Y": 60,
    "7Y": 84,
    "10Y": 120,
    "15Y": 180,
    "20Y": 240,
    "30Y": 360,
}

tenor_order = list(TENOR_MONTHS.keys())
vbma_tenor_order = list(VBMA_TENOR_MONTHS.keys())

def get_bond_tenor(
    issue_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
    type: str,
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

    if type == 'vbma':
        tenor_dates = [
            (
                tenor,
                issue_date + pd.DateOffset(months=months),
                months,
            )
            for tenor, months in VBMA_TENOR_MONTHS.items()
        ]
    elif type == 'vbma_bond_fi':

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
    vbma_bond_fi: pd.DataFrame
    vbma: pd.DataFrame
    holiday_calendar: pd.DataFrame

    def calc_ytm_df_base_on_vbma_bond_fi(self):
        fixed_rate = [float(x) for x in str(self.bond['annual_coupon_rate']).split(";")]
        coupon_change_date = pd.to_datetime(str(self.bond['coupon_change_date']).split(";"), format='mixed')
        min_date = pd.to_datetime(self.vbma_bond_fi.index.min())
        issue_date = max(
            pd.to_datetime(self.bond["issue_date"]),
            min_date
        )
        maturity_date = adjust_following(pd.to_datetime(self.bond['maturity_date']), holidays=self.holiday_calendar)
        bond_tenor = get_bond_tenor(issue_date=issue_date, maturity_date=maturity_date, type='vbma_bond_fi')
        if len(coupon_change_date) > 0 and coupon_change_date.notna().any():
            change_tenor = [get_bond_tenor(issue_date, d, type='vbma_bond_fi') for d in coupon_change_date if pd.notna(d)]
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
                raise ValueError(f"Benchmark does not contain tenor {tenor}")

            value = (rate - benchmark.loc[reset_date, tenor])
            if pd.isna(value):
                value = 0.0
            margins[tenor] = value
        print(f'margin: {margins}')

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
            result.loc[:, columns] = (result.loc[:, columns] + margin)
        return result
        
    def calc_ytm_base_on_vbma(self):
        fixed_rate = [float(x) for x in str(self.bond["annual_coupon_rate"]).split(";")]
        coupon_change_date = pd.to_datetime(str(self.bond["coupon_change_date"]).split(";"),format="mixed")
        issue_date = pd.to_datetime(self.bond["issue_date"])
        maturity_date = adjust_following(pd.to_datetime(self.bond["maturity_date"]), holidays=self.holiday_calendar)
        bond_tenor = get_bond_tenor(issue_date=issue_date, maturity_date=maturity_date, type='vbma')
        if (len(coupon_change_date) > 0 and coupon_change_date.notna().any()):
            change_tenor = [get_bond_tenor(issue_date, d, type='vbma') for d in coupon_change_date if pd.notna(d)]
        else:
            change_tenor = []

        buffer_tenor = [bond_tenor] + change_tenor
        buffer_tenor = sorted(buffer_tenor, key=lambda x: VBMA_TENOR_MONTHS[x])

        if len(buffer_tenor) != len(fixed_rate):
            raise ValueError(
                f"len(buffer_tenor)={len(buffer_tenor)} "
                f"khác len(fixed_rate)={len(fixed_rate)}"
            )

        tenor_rate = sorted(zip(buffer_tenor, fixed_rate), key=lambda x: VBMA_TENOR_MONTHS[x[0]])
        print(tenor_rate)
        vbma = self.vbma.copy()

        valid_vbma_dates = vbma.index[vbma.index <= issue_date]

        if len(valid_vbma_dates) == 0:
            raise ValueError(
                f"No VBMA data available on or before "
                f"issue_date={issue_date}"
            )

        issue_vbma_date = valid_vbma_dates.max()
        vbma_issue = vbma.loc[issue_vbma_date]

        margin_1 = {}
        for tenor, rate in tenor_rate:
            margin_1[tenor] = (rate - vbma_issue[tenor])
        
        # min_date = pd.to_datetime(self.min_date)
        vbma_fi = self.vbma_bond_fi.copy()
        fi_date = vbma_fi.index.min()
        valid_vbma_dates = vbma.index[vbma.index <= fi_date]

        if len(valid_vbma_dates) == 0:
            raise ValueError(
                f"No VBMA data available on or before "
                f"VBMA Bond FI date={fi_date}"
            )

        vbma_min_date = valid_vbma_dates.max()
        vbma_at_min = vbma.loc[vbma_min_date]
        vbma_fi_at_min = vbma_fi.loc[fi_date]
        margin_2 = {}

        for tenor, _ in tenor_rate:
            if tenor not in vbma_at_min.index:
                raise ValueError(
                    f"VBMA does not contain tenor {tenor}"
                )

            if tenor not in vbma_fi_at_min.index:
                raise ValueError(
                    f"VBMA Bond FI does not contain tenor {tenor}"
                )

            margin_2[tenor] = (vbma_fi_at_min[tenor] - vbma_at_min[tenor])

        margin = {}
        for tenor, _ in tenor_rate:
            if tenor not in margin_2:
                raise ValueError(
                    f"Missing Margin 2 for tenor {tenor}"
                )

            value = (margin_1[tenor] - margin_2[tenor])
            if pd.isna(value):
                value = 0.0
            margin[tenor] = value
        print(f'margin: {margin}')
        result = vbma_fi.copy()
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

            margins = margin[tenor]
            result.loc[:, columns] = (result.loc[:, columns] + margins)
        return result
    def calc_ytm_df(self):
        min_date = pd.to_datetime(self.vbma_bond_fi.index.min())
        issue_date = pd.to_datetime(self.bond["issue_date"])
        if min_date <= issue_date:
            ytm_df = self.calc_ytm_df_base_on_vbma_bond_fi()
        elif min_date > issue_date:
            ytm_df = self.calc_ytm_base_on_vbma()
        return ytm_df
    
    # def calc_zyc_df(
    #         self, 
    #         type: str,
    # ):
    
    def calc_zyc_df(self):
        # if type == 'issue_date':
        #     ytm_df = self.calc_ytm_df()
        # elif type == 'value_date':
        #     ytm_df = self.calc_ytm_df_base_on_vbma_bond_fi()
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
