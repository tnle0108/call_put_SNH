#%%
from pathlib import Path
from math import gcd
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, str(Path.cwd().parents[1]))

from src.curve import YieldCurve
from src.daycount import DayCount
from src.multi_hw_tree import BondSpec, CouponDef, ExerciseSpec, MultiCurveHWTree
from src.bond_schedule import CouponSchedule, load_holiday_calendar, adjust_following, build_times, build_times_normal
from src.map_curve import MapCurve

#%%
root = Path.cwd().resolve().parent.parent

BOND_DATA_PATH = root / "datasets" / "raw" / "bonds_placeholder.csv"
CURVE_DATA_PATH = root / "datasets" / "raw" / "Histocopy_FI_ZYC_VND_GD2_family.xlsx"
RAW_PATH = root/'datasets'/'raw'
_HOLIDAY = Path.cwd().parents[1] / "datasets" / "holidays"

REPORT_DATE = pd.to_datetime("2026-06-03")

DISC_CONVENTION = "actactisda"
REF_CONVENTION = "actactisda"
COUP_CONVENTION = "act365"
#%%


bond_df = pd.read_csv(BOND_DATA_PATH)
bond_df["issue_date"] = pd.to_datetime(bond_df["issue_date"])
bond_df["maturity_date"] = pd.to_datetime(bond_df["maturity_date"])
groups = ["LB_G1", "LB_G2", "LB_G3", "LB_G4", "NBFI", "FB"]

dfs = {}
curves = {}
ref_curves = {}

daycount = DayCount.get(DISC_CONVENTION)
    
#%%
holiday_calendar = load_holiday_calendar(_HOLIDAY)

coupon_schedule_df = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country = 'vnd'
).build_coupon_schedule_df()

coupon_map = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country ='vnd'
).build_coupon_map_from_schedule(bond_df.iloc[0])


#%%

results = []

for _, row in bond_df.iterrows():
    bond_id = str(row["bond_id"])
    issue_date = pd.to_datetime(row["issue_date"])
    ref_curve_name = (None if pd.isna(row["ref_curve"]) else str(row["ref_curve"]).strip().lower())
    print(ref_curve_name)
    style = str(row['style']).strip().lower()
    maturity_date = pd.to_datetime(row['maturity_date'])

    group = str(row["group"]).strip() if "group" in row and pd.notna(row["group"]) else None

    if pd.notna(row["call_exercise_dates"]) and str(row["call_exercise_dates"]).strip():
        call_dates = [pd.to_datetime(x.strip(), format = 'mixed') for x in str(row["call_exercise_dates"]).split(";")]
        call_strikes = [float(x) for x in str(row["call_strike"]).split(";")]
    else:
        call_dates = []
        call_strikes = []
    
    if pd.notna(row["put_exercise_dates"]) and str(row["put_exercise_dates"]).strip():
        put_dates = [pd.to_datetime(x.strip(), format = 'mixed') for x in str(row["put_exercise_dates"]).split(";")]
        put_strikes = [float(x) for x in str(row["put_strike"]).split(";")]
    else:
        put_dates = []
        put_strikes = []


    def to_step(date, times):
        idx = np.where(times == date)[0]
        if len(idx) == 0:
            raise ValueError(f"{date} không có trong times")
        return int(idx[0])

    call = {to_step(d, times_normal): s
            for d, s in zip(call_dates, call_strikes)}

    put = {to_step(d, times_normal): s
        for d, s in zip(put_dates, put_strikes)}

    
    print('put_dates:', put_dates)
    print('call_dates:', call_dates)

    disc_curve = MapCurve(
        rpd=REPORT_DATE,
        curve_folder=str(RAW_PATH),
        convention=DISC_CONVENTION,
    ).map_disc_curve(group)

    if ref_curve_name is not None:
        ref_curve = MapCurve(
            rpd = REPORT_DATE,
            curve_folder=str(RAW_PATH),
            convention=REF_CONVENTION,
        ).map_ref_curve(ref_curve_name)
    else:
        ref_curve = None

    coupon_dates = (
        coupon_schedule_df.loc[
            coupon_schedule_df["bond_id"] == row["bond_id"],
            "pay_date",
        ]
        .sort_values()
        .tolist()
    )

    times=build_times(
        DISC_CONVENTION,
        issue_date,
        coupon_dates,
        maturity_date,
        call_dates,
        put_dates,
    )

    times_normal = build_times_normal(
        issue_date,
        coupon_dates,
        maturity_date,
        call_dates,
        put_dates,
    )

    def to_step(date, times):
        idx = np.where(times == date)[0]
        if len(idx) == 0:
            raise ValueError(f"{date} không có trong times")
        return int(idx[0])

    call = {to_step(d, times_normal): s
            for d, s in zip(call_dates, call_strikes)}

    put = {to_step(d, times_normal): s
        for d, s in zip(put_dates, put_strikes)}


    tree = MultiCurveHWTree(
        a_r=0.03,
        sigma_r=0.015,
        a_L=0.02,
        sigma_L=0.018,
        rho=0.20,
        disc_curve=disc_curve,
        ref_curve=ref_curve,
        times=times,
        times_normal = times_normal,
    )

    maturity_step = len(times) - 1
    face = float(row["face"])
    coupon_accrual = float(row["coupon_accrual"])
    coupon_map = CouponSchedule(
        df=bond_df,
        holiday_calendar=holiday_calendar,
        country ='vnd'
    ).build_coupon_map_from_schedule(row)
    bond = BondSpec(
        face=face,
        maturity_step=maturity_step,
        coupons=coupon_map,
    )
    exercise = ExerciseSpec(call=call, put=put)
    full_price = tree.price(bond, exercise)
    straight_price = tree.price(bond)
    
    results.append({
        "bond_id": bond_id,
        "group": group,
        "ref_curve": ref_curve_name,
        "style": style,
        "maturity_step": maturity_step,
        "full_price": full_price,
        "straight_price": straight_price,
        "option_price": full_price - straight_price,
    })

results_df = pd.DataFrame(results)
results_df

#%%
