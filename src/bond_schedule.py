#%%

import pandas as pd
import numpy as np
import sys
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[0]))

from src.daycount import DayCount
from src.multi_hw_tree import CouponDef

#%%
# _PACKAGE_SRC = Path(__file__).resolve().parents[1]
# _HOLIDAY = _PACKAGE_SRC / "datasets" / "holidays"
extra_working_days = {
    pd.Timestamp("2026-01-10"),
    pd.Timestamp("2024-05-04"),
    pd.Timestamp("2025-04-26"),
}


def load_holiday_calendar(folder):
    holiday_dict = {}

    for file in Path(folder).glob("*.csv"):
        country = file.stem

        df = pd.read_csv(file)

        holidays = set(
            pd.to_datetime(df["Date"]).dt.normalize()
        )

        holiday_dict[country] = holidays

    return holiday_dict




def adjust_following(date, holidays):
    date = pd.Timestamp(date).normalize()

    while True:
        if date in holidays:
            date += timedelta(days=1)
            continue

        if date.weekday() >= 5 and date not in extra_working_days:
            date += timedelta(days=1)
            continue

        break

    return date


def build_times(convention, issue_date, coupon_dates, call_dates = None, put_dates = None):
    call_dates = call_dates or []
    put_dates = put_dates or []
    dates=[issue_date, *coupon_dates, *call_dates, *put_dates]
    dates=sorted(set(dates))
    times=[]

    for d in dates:
        t=DayCount.get(convention).yearfrac(
            np.array([issue_date],dtype="datetime64[D]"),
            np.array([d],dtype="datetime64[D]")
        )[0]
        times.append(t)

    return np.array(times)

def build_times_normal(issue_date, coupon_dates, call_dates = None, put_dates = None):
    call_dates = call_dates or []
    put_dates = put_dates or []
    dates=[issue_date, *coupon_dates, *call_dates, *put_dates]
    dates=sorted(set(dates))
    times=[]

    for d in dates:
        times.append(d)
    return np.array(times)

@dataclass
class CouponSchedule:
    df: pd.DataFrame
    holiday_calendar: dict
    country: str

    def build_coupon_schedule_df(self) -> pd.DataFrame:
        rows = []
        for _, bond in self.df.iterrows():
            issue_date = pd.to_datetime(bond["issue_date"])
            maturity_date = pd.to_datetime(bond["maturity_date"])

            accrual = float(bond["coupon_accrual"])
            months = int(round(accrual * 12))

            pay_dates = pd.date_range(
                start=issue_date,
                end=maturity_date,
                freq=pd.DateOffset(months=months),
            )[1:]       

            holidays = self.holiday_calendar.get(self.country, set())
            pay_dates = [adjust_following(d, holidays) for d in pay_dates]

            margin_dates = []
            margin_values = []

            if pd.notna(bond["margin_date"]) and str(bond["margin_date"]).strip():
                margin_dates = [pd.to_datetime(x.strip(), format="mixed") for x in str(bond["margin_date"]).split(";")]
                margin_values = [float(x) for x in str(bond["margin"]).split(";")]

            is_fixed = pd.notna(bond["annual_coupon_rate"])

            for step, pay_date in enumerate(pay_dates, start=1):
                if is_fixed:
                    margin = np.nan
                else:
                    margin = margin_values[0]
                    for d, m in zip(margin_dates, margin_values):
                        if pay_date >= d:
                            margin = m
                rows.append(
                    {
                        "bond_id": bond["bond_id"],
                        "step": step,
                        "pay_date": pay_date,
                        "accrual": accrual,
                        "coupon_type": "fixed" if is_fixed else "float",
                        "fixed_rate": (
                            bond["annual_coupon_rate"]
                            if is_fixed
                            else np.nan
                        ),
                        "ref_curve": bond["ref_curve"],
                        "ref_tenor": bond["ref_tenor"],
                        "margin": margin,
                        "floor": bond["floor"],
                        "cap": bond["cap"],
                    }
                )

        return pd.DataFrame(rows)


    def build_coupon_map_from_schedule(self, row: pd.Series) -> dict[int, CouponDef]:
        coupon_schedule_df = self.build_coupon_schedule_df()
        bond_id = str(row["bond_id"])
        sub = coupon_schedule_df[coupon_schedule_df["bond_id"] == bond_id].sort_values("step")

        coupon_map = {}

        for _, s in sub.iterrows():
            step = int(s["step"])
            accrual = float(s["accrual"])  # vd 0.25, 0.5, 1.0
            coupon_type = str(s["coupon_type"]).strip().lower()

            if coupon_type == "fixed":
                coupon_map[step] = CouponDef(
                    accrual=accrual,
                    fixed_rate=float(s["fixed_rate"]),
                )
            elif coupon_type == "float":
                coupon_map[step] = CouponDef(
                    accrual=accrual,
                    fixed_rate=None,
                    margin=float(s["margin"]),
                    ref_tenor=float(s["ref_tenor"]),#debug
                    floor=s["floor"] if pd.notna(s["floor"]) else None,
                    cap=s["cap"] if pd.notna(s["cap"]) else None,
                )
            else:
                raise ValueError(f"Unsupported coupon_type={coupon_type}")

        return coupon_map
