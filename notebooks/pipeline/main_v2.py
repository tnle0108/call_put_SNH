#%%
import pandas as pd
import sys
import os
import json
import time

from pathlib import Path
print(sys.path.insert(0, str(Path.cwd().parents[1])))

from callput import (
    CallPutTree,
    CurveLeg,
    compile_bond,
)
from src.map_curve import MapCurve
from src.bond_schedule import CouponSchedule, load_holiday_calendar, build, adjust_following
#%%
root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH = os.path.join(root, 'datasets', 'raw')
CURVE_FOLDER_PATH = os.path.join(root, 'datasets', 'curve')
HOLIDAY_FOLDER_PATH = Path.cwd().parents[1] / "datasets" / "holidays"
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

VALUE_DATE = pd.to_datetime('2026-03-03')

DISC_CONVENTION = 'ACT/365'
REF_CONVENTION = 'ACT/365'
COUP_CONVENTION = 'actactisda'

DISC_NAME = 'vbma_bond_fi'
REF_NAME = 'SOB4'

STEP_DAYS = float(21)
MIN_STEP_DAYS = float(3)
RHO = 0.02
#%%
bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'bonds_placeholder - Copy.csv'))
disc_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{DISC_NAME}.csv'))
ref_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{REF_NAME}.csv'))


#%%
def legs(
        disc_curve,
        ref_curve,
        a_r, sigma_r,
        a_L, sigma_L,
        disc_conv, ref_conv,
        disc_bump=0.0,
        ref_bump=0.0,
):
    disc = disc_curve.shifted(disc_bump)
    if ref_curve is None:
        return CurveLeg(disc, a_r, sigma_r, disc_conv), None
    ref = ref_curve.shifted(ref_bump)
    return (
        CurveLeg(disc, a_r, sigma_r, disc_conv),
        CurveLeg(ref, a_L, sigma_L, ref_conv),
    )

def make_tree(sched, step_days=STEP_DAYS, **kw):
    bond = compile_bond(sched, step_days=step_days, min_step=MIN_STEP_DAYS)
    disc, ref = legs(**kw)
    if ref is None:
        return bond, CallPutTree.single_curve(bond, disc)
    return bond, CallPutTree.multi_curve(bond, disc, ref, rho=RHO)

#%%

disc_curve = MapCurve(
    rpd = VALUE_DATE,
    curve_folder = CURVE_FOLDER_PATH,
    convention = DISC_CONVENTION
).map_curve(DISC_NAME)

with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
    hw_params = json.load(f)

a_r = hw_params[DISC_NAME]['a']
sigma_r = hw_params[DISC_NAME]['sigma']

disc_leg = CurveLeg(
    curve = disc_curve,
    a = a_r,
    sigma = sigma_r,
    dcc = DISC_CONVENTION
)

holiday_calendar = load_holiday_calendar(HOLIDAY_FOLDER_PATH)

coupon_schedule_df = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country = 'vnd'
).build_coupon_schedule_df()


#%%
results = []

for _, row in bond_df.iterrows():
    bond_id = str(row['bond_id'])
    print(bond_id)
    issue_date = pd.to_datetime(row['issue_date'])
    ref_curve_name = (None if pd.isna(row['ref_curve']) else str(row['ref_curve']).strip().lower())
    maturity_date = adjust_following(pd.to_datetime(row['maturity_date']), holiday_calendar)
    coupon_accrual = float(row['coupon_accrual'])
    face_value = float(row['face'])

    if pd.notna(row["call_exercise_dates"]) and str(row["call_exercise_dates"]).strip():
        call_dates = [pd.to_datetime(x.strip(), format='mixed') for x in str(row["call_exercise_dates"]).split(";")]
        call_strikes = [float(x)/face_value for x in str(row["call_strike"]).split(";")]
    else:
        call_dates = []
        call_strikes = []

    if pd.notna(row["put_exercise_dates"]) and str(row["put_exercise_dates"]).strip():
        put_dates = [pd.to_datetime(x.strip(), format='mixed') for x in str(row["put_exercise_dates"]).split(";")]
        put_strikes = [float(x)/face_value for x in str(row["put_strike"]).split(";")]
    else:
        put_dates = []
        put_strikes = []

    call_df = pd.DataFrame({
        "call_date": pd.Series(call_dates, dtype="datetime64[ns]"),
        "call_strike": pd.Series(call_strikes, dtype="float64"),
    })
    put_df = pd.DataFrame({
        "put_date": pd.Series(put_dates, dtype="datetime64[ns]"),
        "put_strike": pd.Series(put_strikes, dtype="float64"),
    })

    if ref_curve_name is not None:
        ref_curve = MapCurve(
            rpd=VALUE_DATE,
            curve_folder=str(CURVE_FOLDER_PATH),
            convention=REF_CONVENTION,
        ).map_curve(ref_curve_name)
        a_L = hw_params[REF_NAME]['a']
        sigma_L = hw_params[REF_NAME]['sigma']
        reset_dates = CouponSchedule(
            df=bond_df[bond_df['bond_id'] == row['bond_id']],
            holiday_calendar=holiday_calendar,
            country='vnd'
        ).bulid_reset_schedule()
        fixed_rate = None
    else:
        ref_curve = None
        fixed_rate = float(row['annual_coupon_rate'])
        reset_dates = []
        a_L = 0.0
        sigma_L = 0.0

    # -------------------------------------------------------------
    def build_sched(reading, apply_floor=True, apply_cap=True, _row=row, _call_df=call_df, _put_df=put_df):
        return build(
            reading=reading,
            rpd=VALUE_DATE,
            face=face_value,
            maturity_date=maturity_date,
            coupon_accrual=coupon_accrual,
            coupon_schedule_df=coupon_schedule_df[coupon_schedule_df['bond_id'] == _row['bond_id']],
            ref_dates=reset_dates,
            call_df=_call_df,
            put_df=_put_df,
            fixed_rate=fixed_rate,
            ref_curve_name=ref_curve_name,
            curve_folder=str(CURVE_FOLDER_PATH),
            ref_convention=REF_CONVENTION,
            apply_floor=apply_floor,
            apply_cap=apply_cap,
        )

    def make_tree_for_bond(sched, step_days=STEP_DAYS, **bump_kw):
        return make_tree(
            sched,
            step_days=step_days,
            disc_curve=disc_curve,
            ref_curve=ref_curve,
            a_r=a_r,
            sigma_r=sigma_r,
            a_L=a_L,
            sigma_L=sigma_L,
            disc_conv=DISC_CONVENTION,
            ref_conv=REF_CONVENTION,
            **bump_kw,
        )
    # -------------------------------------------------------------

    sched = build_sched('advance')
    bond, tree = make_tree_for_bond(sched)
    full_price = tree.price()


    step = bond.step_days()
    for w in bond.warnings:
        print(f'warning:{w}')

    started = time.perf_counter()
    parts = tree.decompose()
    elapsed = time.perf_counter() - started

    for reading in ('advance', 'arrears'):
        sched_r = build_sched(reading)
        sched_r.call = {}
        _, alt = make_tree_for_bond(sched_r)
        parts = alt.decompose()

    try:
        sched_arrears = build_sched('arrears')
        _, alt = make_tree_for_bond(sched_arrears)
        alt.price()
    except ValueError as exc:
        print(f"\nWith the call schedule the arrears reading is refused:\n  {exc}")

    sched_straight = build_sched('advance', apply_floor=False, apply_cap=False)
    sched_straight.call = {}
    sched_straight.put = {}
    _, tree_straight = make_tree_for_bond(sched_straight)
    straight_price = tree_straight.price()

    results.append({
        "bond_id": bond_id,
        "full_price": full_price,
        "straight_price": straight_price,
        "diff": full_price - straight_price,
    })

    # for step_days in (56, 42, 28, 21, 14):
    #     _, conv = make_tree_for_bond(sched, step_days=step_days)
    #     d = conv.decompose()

    # base = tree.price()
    # for bump in (-0.02, -0.01, 0.01, 0.02):
    #     _, shocked = make_tree_for_bond(sched, disc_bump=bump, ref_bump=bump)
    #     price = shocked.price()    
#%%


