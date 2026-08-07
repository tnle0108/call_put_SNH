#%%

import pandas as pd
import numpy as np
import sys
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path.cwd().parents[0]))

from src.daycount import DayCount
from src.multi_hw_tree import CouponDef
from callput import (
    BondSchedule,
    CouponPeriod,
)

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


def build_times(convention, issue_date, coupon_dates, maturity_date, call_dates = None, put_dates = None):
    call_dates = call_dates or []
    put_dates = put_dates or []
    dates=[issue_date,maturity_date, *coupon_dates, *call_dates, *put_dates]
    dates=sorted(set(dates))
    times=[]

    for d in dates:
        t=DayCount.get(convention).yearfrac(
            np.array([issue_date],dtype="datetime64[D]"),
            np.array([d],dtype="datetime64[D]")
        )[0]
        times.append(t)

    return np.array(times)

def build_times_normal(issue_date, coupon_dates,maturity_date, call_dates = None, put_dates = None):
    call_dates = call_dates or []
    put_dates = put_dates or []
    dates=[issue_date,maturity_date, *coupon_dates, *call_dates, *put_dates]
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
            print(bond['bond_id'])
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
                    print(margin_values)
                    margin = margin_values[0]
                    for i, d in enumerate(margin_dates):
                        if pay_date >= d:
                            margin = margin_values[i + 1]
                        else:
                            break
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
                    ref_tenor=str(s["ref_tenor"]),#debug
                    floor=s["floor"] if pd.notna(s["floor"]) else None,
                    cap=s["cap"] if pd.notna(s["cap"]) else None,
                )
            else:
                raise ValueError(f"Unsupported coupon_type={coupon_type}")

        return coupon_map
    def bulid_reset_schedule(self):
        rows = []
        for _, bond in self.df.iterrows():
            issue_date = pd.to_datetime(bond["issue_date"])
            maturity_date = pd.to_datetime(bond["maturity_date"])
            months = int(bond['ref_tenor'][:-1])
            reset_dates = pd.date_range(
                start = issue_date,
                end = maturity_date,
                freq = pd.DateOffset(months = months),
            )
            holidays = self.holiday_calendar.get(self.country, set())
            reset_dates = [
                pd.Timestamp(adjust_following(d, holidays))
                for d in reset_dates
            ]

            if reset_dates and reset_dates[-1] == maturity_date:
                reset_dates = reset_dates[:-1]

            rows.append(reset_dates)

        return rows

def days(d:date, start_date:date) -> int:
    return(d-start_date).days

def build(
        reading: str,
        face: float,
        maturity_date: pd.Timestamp,
        coupon_accrual: float,
        coupon_schedule_df: pd.DataFrame,
        ref_dates: list[date] | None = None,
        call_df: pd.DataFrame | None = None,
        put_df: pd.DataFrame | None = None,
        fixed_rate: float | None = None,
) -> BondSchedule:
    pays = coupon_schedule_df["pay_date"].sort_values().tolist()
    months = round(coupon_accrual * 12)
    first_start = pays[0] - relativedelta(months=months)
    starts = [first_start] + pays[:-1]
    resets = ref_dates or []

    is_floating = fixed_rate is None
    if is_floating:
        margin_map = coupon_schedule_df.set_index('pay_date')['margin']
        floor_map = coupon_schedule_df.set_index('pay_date')['floor']

    periods = []
    for pay, start in zip(pays, starts):
        accrual = (pay - start).days / 365.0

        # fixed-rate bond, hoặc kỳ đầu của floating bond chưa có fixing
        if not is_floating or pay <= resets[0]:
            periods.append(
                CouponPeriod(
                    pay_day=days(pay),
                    accrual=accrual,
                    accrual_start_day=days(start),
                    fixed_rate=fixed_rate,
                )
            )
            continue

        if reading == 'advance':
            fixing = max(r for r in resets if r < pay)
        else:
            fixing = max(r for r in resets if r <= pay)

        periods.append(
            CouponPeriod(
                pay_day=days(pay),
                accrual=accrual,
                accrual_start_day=days(start),
                fixing_day=days(fixing),
                ref_tenor_days=365,
                margin=margin_map[pay],
                floor=floor_map[pay],
            )
        )
        call = (
            {days(d): s for d, s in zip(call_df['call_date'], call_df['call_strike'])}
            if call_df is not None and not call_df.empty
            else {}
        )
        put = (
            {days(d): s for d, s in zip(put_df['put_date'], put_df['put_strike'])}
            if put_df is not None and not put_df.empty
            else {}
        )
    return BondSchedule(
        face = face,
        maturity_day = days(maturity_date),
        periods=periods,
        call = call,
        put = put,
    )

