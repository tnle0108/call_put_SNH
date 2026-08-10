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

from src.map_curve import MapCurve


def tenor_to_years(tenor: str) -> float:
    tenor = str(tenor).strip().upper()
    if tenor.endswith("Y"):
        return int(tenor[:-1])
    elif tenor.endswith("M"):
        return int(tenor[:-1]) / 12.0
    elif tenor.endswith("W"):
        return int(tenor[:-1]) / 52.0
    elif tenor.endswith("N"):
        return 1 / 365.0
    else:
        raise ValueError(f"Unsupported tenor: {tenor}")


def get_known_rate(
    ref_curve_name: str,
    ref_tenor: str,
    fixing_date: pd.Timestamp,
    margin: float,
    curve_folder: str,
    convention: str,
) -> float:
    hist_curve = MapCurve(
        rpd=fixing_date,
        curve_folder=curve_folder,
        convention=convention,
    ).map_curve(ref_curve_name)

    t = tenor_to_years(ref_tenor)
    ref_rate = hist_curve.zero_rate(t)
    return float(ref_rate) + float(margin)

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

def nan_to_none(x):
    return None if pd.isna(x) else float(x)

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
                start=issue_date, end=maturity_date, freq=pd.DateOffset(months=months),
            )[1:]
            holidays = self.holiday_calendar.get(self.country, set())
            pay_dates = [adjust_following(d, holidays) for d in pay_dates]

            is_fixed = pd.notna(bond["annual_coupon_rate"])

            margin_dates_raw = str(bond["margin_date"]) if pd.notna(bond["margin_date"]) and str(bond["margin_date"]).strip() else None
            margin_values_raw = str(bond["margin"]) if pd.notna(bond["margin"]) and str(bond["margin"]).strip() else None

            for step, pay_date in enumerate(pay_dates, start=1):
                rows.append(
                    {
                        "bond_id": bond["bond_id"],
                        "step": step,
                        "pay_date": pay_date,
                        "accrual": accrual,
                        "coupon_type": "fixed" if is_fixed else "float",
                        "fixed_rate": bond["annual_coupon_rate"] if is_fixed else np.nan,
                        "ref_curve": bond["ref_curve"],
                        "ref_tenor": bond["ref_tenor"],
                        "margin_dates_raw": margin_dates_raw,
                        "margin_values_raw": margin_values_raw,
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

            return reset_dates

        return []

def resolve_margin(margin_dates_raw, margin_values_raw, ref_date):
    if margin_values_raw is None or pd.isna(margin_values_raw) or not str(margin_values_raw).strip():
        return None
    values = [float(x) for x in str(margin_values_raw).split(";")]
    if margin_dates_raw is None or pd.isna(margin_dates_raw) or not str(margin_dates_raw).strip():
        return values[0]
    mdates = [pd.to_datetime(x.strip(), format="mixed") for x in str(margin_dates_raw).split(";")]
    if len(values) != len(mdates) + 1:
        raise ValueError(
            f"margin có {len(values)} giá trị nhưng margin_date có {len(mdates)} mốc"
        )
    margin = values[0]
    for i, d in enumerate(mdates):
        if ref_date >= d:
            margin = values[i + 1]
        else:
            break
    return margin

def days(d:date, start_date:date) -> int:
    return(d-start_date).days
def build(
        reading: str,
        rpd: date,
        face: float,
        maturity_date: pd.Timestamp,
        coupon_accrual: float,
        coupon_schedule_df: pd.DataFrame,
        ref_dates: list[date] | None = None,
        call_df: pd.DataFrame | None = None,
        put_df: pd.DataFrame | None = None,
        fixed_rate: float | None = None,
        ref_curve_name: str | None = None,
        curve_folder: str | None = None,
        ref_convention: str | None = None,
        apply_floor: bool = True,
        apply_cap: bool = True,
) -> BondSchedule:
    pays_all = coupon_schedule_df["pay_date"].sort_values().tolist()
    months = round(coupon_accrual * 12)
    first_start_all = pays_all[0] - relativedelta(months=months)
    starts_all = [first_start_all] + pays_all[:-1]

    future_pairs = [(s, p) for s, p in zip(starts_all, pays_all) if p > rpd]
    if not future_pairs:
        raise ValueError(f"No future coupon periods after rpd={rpd}")
    starts, pays = map(list, zip(*future_pairs))

    resets = ref_dates or []

    is_floating = fixed_rate is None
    if is_floating:
        margin_dates_map = coupon_schedule_df.set_index('pay_date')['margin_dates_raw']
        margin_values_map = coupon_schedule_df.set_index('pay_date')['margin_values_raw']
        floor_map = coupon_schedule_df.set_index('pay_date')['floor']
        cap_map = coupon_schedule_df.set_index('pay_date')['cap']
        tenor_map = coupon_schedule_df.set_index('pay_date')['ref_tenor']

    periods = []
    for pay, start in zip(pays, starts):
        accrual = (pay - start).days / 365.0

        if not is_floating:
            periods.append(
                CouponPeriod(
                    pay_day=days(pay, rpd),
                    accrual=accrual,
                    accrual_start_day=days(start, rpd),
                    fixed_rate=fixed_rate,
                    fixing_day=None,
                    margin=0.0,
                    ref_tenor_days=365,
                    floor=None,
                    cap=None,
                )
            )
            continue

        if reading == 'advance':
            candidates = [r for r in resets if r < pay]
        else:
            candidates = [r for r in resets if r <= pay]

        if not candidates:
            raise ValueError(
                f"No fixing date found for payment date {pay} (reading={reading})"
            )
        fixing = max(candidates)

        if fixing <= rpd:
            if curve_folder is None or ref_curve_name is None or ref_convention is None:
                raise ValueError(
                    f"Period with fixing {fixing} <= rpd {rpd} requires "
                    f"curve_folder/ref_curve_name/ref_convention to look up the known rate"
                )
            margin_known = resolve_margin(margin_dates_map[pay], margin_values_map[pay], fixing)
            known_rate = get_known_rate(
                ref_curve_name=ref_curve_name,
                ref_tenor=tenor_map[pay],
                fixing_date=fixing,
                margin=margin_known,
                curve_folder=curve_folder,
                convention=ref_convention,
            )
            periods.append(
                CouponPeriod(
                    pay_day=days(pay, rpd),
                    accrual=accrual,
                    accrual_start_day=days(start, rpd),
                    fixed_rate=known_rate,
                    fixing_day=None,
                    margin=0.0,
                    ref_tenor_days=365,
                    floor=None,
                    cap=None,
                )
            )
            continue

        margin = resolve_margin(margin_dates_map[pay], margin_values_map[pay], fixing)
        periods.append(
            CouponPeriod(
                pay_day=days(pay, rpd),
                accrual=accrual,
                accrual_start_day=days(start, rpd),
                fixing_day=days(fixing, rpd),
                ref_tenor_days=365,
                margin=margin if margin is not None else 0.0,
                floor=nan_to_none(floor_map[pay]) if apply_floor else None,
                cap=nan_to_none(cap_map[pay]) if apply_cap else None,
            )
        )

    dropped_calls = call_df[call_df['call_date'] <= rpd] if call_df is not None and not call_df.empty else None
    if dropped_calls is not None and not dropped_calls.empty:
        print(f"[{None}] Warning: {len(dropped_calls)} call date(s) already in the past, dropped: {dropped_calls['call_date'].tolist()}")

    dropped_puts = put_df[put_df['put_date'] <= rpd] if put_df is not None and not put_df.empty else None
    if dropped_puts is not None and not dropped_puts.empty:
        print(f"Warning: {len(dropped_puts)} put date(s) already in the past, dropped: {dropped_puts['put_date'].tolist()}")

    call = (
        {days(d, rpd): s for d, s in zip(call_df['call_date'], call_df['call_strike']) if d > rpd}
        if call_df is not None and not call_df.empty
        else {}
    )
    put = (
        {days(d, rpd): s for d, s in zip(put_df['put_date'], put_df['put_strike']) if d > rpd}
        if put_df is not None and not put_df.empty
        else {}
    )

    return BondSchedule(
        face=face,
        maturity_day=days(maturity_date, rpd),
        periods=periods,
        call=call,
        put=put,
    )