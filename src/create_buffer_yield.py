"""Bảng kỳ hạn chuẩn và bootstrap đường zero từ báo giá YTM.

Tên file là di sản: lớp ``BufferYTM`` dựng đường Tier 2 bằng cách đắp buffer
lên VBMA đã bị xoá, vì phần bù tăng vốn nay là ``delta`` trên ``x(t)`` dò ngược
từ giá quan sát (``src/tier2_spread.py``).
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
    """Bootstrap báo giá YTM của một phân nhóm TCPH thành đường zero.

    Đường chiết khấu luôn là đường của phân nhóm, kể cả với trái phiếu tăng
    vốn — phần bù tăng vốn vào sau, ở không gian ``x(t)``.
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
