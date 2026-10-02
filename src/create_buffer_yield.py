"""
Standard tenor grid and bootstrapping of zero curves from YTM quotes.

The file name is a legacy: the ``BufferYTM`` class, which built the Tier 2 curve by adding a buffer
on top of VBMA, has been deleted, because the Tier 2 (capital bond) premium is now a ``delta`` on
``x(t)`` backed out from observed prices (``src/tier2_spread.py``).
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0]))

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

def calc_zyc_df(ytm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap the YTM quotes of one issuer group (TCPH) into a zero curve.

    The discount curve is always the issuer group's curve, even for Tier 2 capital bonds: the Tier
    2 premium comes in later, in ``x(t)`` space.

    Args:
        ytm_df (pd.DataFrame): YTM quotes, one row per date (the index), passed to BenchmarkCurve
            "FI ZYC VND" as benchmark prices.

    Returns:
        pd.DataFrame: Zero rates, one row per date (index named "Date") and one column per tenor
        in TENOR_MONTHS order.
    """
    order = tenor_order
    benchmark_curve = BenchmarkCurve(
        curve_name="FI ZYC VND",
        benchmark_price=ytm_df,
    )
    zyc_df_list = []
    for rpd in ytm_df.index:
        zyc_df_long = benchmark_curve.print_curve(rpd)
        zyc_df_wide = (
            zyc_df_long
            .loc[order, "value"]
            .rename(rpd)
        )
        zyc_df_list.append(zyc_df_wide)
    zyc_df = pd.DataFrame(zyc_df_list)
    zyc_df.index.name = "Date"
    return zyc_df
