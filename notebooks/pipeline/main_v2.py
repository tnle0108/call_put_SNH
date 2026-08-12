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
from src.shock import ShockScenario
from src.calc_rho import calc_rho

from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.curve.curvenode import CurveNode

#%%
root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH    = os.path.join(root, 'datasets', 'raw')
CURVE_FOLDER_PATH   = os.path.join(root, 'datasets', 'curve')
HOLIDAY_FOLDER_PATH = Path.cwd().parents[1] / "datasets" / "holidays"
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

# VALUE_DATE = pd.to_datetime('2024-12-18')

DISC_CONVENTION = 'ACT/365'
REF_CONVENTION  = 'ACT/365'

DISC_NAME   = 'vbma_bond_fi'

STEP_DAYS       = float(21)
MIN_STEP_DAYS   = float(3)
FIXING_LAG_DAYS = float(7)
MIN_DATE = pd.to_datetime("2024-06-03")
# RHO             = 0.02
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

def make_tree(sched, rho_param, step_days=STEP_DAYS, **kw):
    bond = compile_bond(sched, step_days=step_days, min_step=MIN_STEP_DAYS)
    disc, ref = legs(**kw)
    if ref is None:
        return bond, CallPutTree.single_curve(bond, disc)
    return bond, CallPutTree.multi_curve(bond, disc, ref, rho=rho_param)

#%%
bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'bond placeholder.csv'))
holiday_calendar = load_holiday_calendar(HOLIDAY_FOLDER_PATH)
coupon_schedule_df = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country = 'vnd'
).build_coupon_schedule_df()

#%%

shocks_list = ["0","1", "2", "3", "4", "5", "6"]
bond_results = []

for shock in shocks_list:

    print("*" * 30 + "NEW_SHOCK" + "*"*30)

    disc_df_raw = pd.read_csv(os.path.join(CURVE_FOLDER_PATH,f'{DISC_NAME}.csv'), index_col=0)
    disc_df_raw.index = pd.to_datetime(disc_df_raw.index)
    disc_df_raw = disc_df_raw.sort_index()

    if shock == "0":
        disc_df = disc_df_raw
    else:
        disc_df = ShockScenario(shock_type=shock, df = disc_df_raw).create_shock_df()
    disc_df.to_csv(os.path.join(CURVE_FOLDER_PATH, f'{DISC_NAME}_shocked_{shock}.csv'))

    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        hw_params = json.load(f)

    shocked_disc_curve_name = f"{DISC_NAME}_shocked_{shock}"

    shocked_disc_curve_params = hw_params.get(shocked_disc_curve_name, {})

    if "a" in shocked_disc_curve_params and "sigma" in shocked_disc_curve_params:
        print(f"Hull-White parameters already exist for {shocked_disc_curve_name}.")
        print(f"a     = {shocked_disc_curve_params['a']}")
        print(f"sigma = {shocked_disc_curve_params['sigma']}")

    else:
        print(f"Hull-White parameters not found for {shocked_disc_curve_name}.")
        print("Running Hull-White calibration...")
        hw = HullWhite.get(shocked_disc_curve_name)

        result = hw.calibrate(
            shocked_disc_curve_name,
            method="kfmh",
            save=True,
        )
        
        with open(HULLWHITE_FILE_PATH, "r", encoding="utf-8") as f:
            hw_params = json.load(f)

        shocked_disc_curve_params = hw_params[shocked_disc_curve_name]

    a_r = shocked_disc_curve_params["a"]
    sigma_r = shocked_disc_curve_params["sigma"]

    for _, row in bond_df.iterrows():
        bond_id = str(row['bond_id'])
        print("="*80)
        print(bond_id)
        issue_date = pd.to_datetime(row['issue_date'])
        VALUE_DATE = max(issue_date, MIN_DATE)
        disc_curve = MapCurve(
            rpd = VALUE_DATE,
            df = disc_df,
            convention = DISC_CONVENTION
        ).map_curve()
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
            ref_df_raw = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{ref_curve_name}.csv'), index_col = 0)

            ref_df_raw.index = pd.to_datetime(ref_df_raw.index)
            ref_df_raw = ref_df_raw.sort_index()

            if shock == "0":
                ref_df = ref_df_raw
            else:
                ref_df = ShockScenario(shock_type=shock, df = ref_df_raw).create_shock_df()
            ref_df.to_csv(os.path.join(CURVE_FOLDER_PATH, f'{ref_curve_name}_shocked_{shock}.csv'))

            with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
                hw_params = json.load(f)

            shocked_ref_curve_name = f"{ref_curve_name}_shocked_{shock}"

            shocked_ref_curve_params = hw_params.get(shocked_ref_curve_name, {})

            if "a" in shocked_ref_curve_params and "sigma" in shocked_ref_curve_params:
                print(f"Hull-White parameters already exist for {shocked_ref_curve_name}.")
                print(f"a     = {shocked_ref_curve_params['a']}")
                print(f"sigma = {shocked_ref_curve_params['sigma']}")

            else:
                print(f"Hull-White parameters not found for {shocked_ref_curve_name}.")
                print("Running Hull-White calibration...")

                hw = HullWhite.get(shocked_ref_curve_name)

                result = hw.calibrate(
                    shocked_ref_curve_name,
                    method="kfmh",
                    save=True,
                )
                
                with open(HULLWHITE_FILE_PATH, "r", encoding="utf-8") as f:
                    hw_params = json.load(f)

                shocked_ref_curve_params = hw_params[shocked_ref_curve_name]

            a_L = shocked_ref_curve_params["a"]
            sigma_L = shocked_ref_curve_params["sigma"]

            ref_curve = MapCurve(
                rpd=VALUE_DATE,
                df = ref_df,
                convention=REF_CONVENTION,
            ).map_curve()

            reset_dates = CouponSchedule(
                df=bond_df[bond_df['bond_id'] == row['bond_id']],
                holiday_calendar=holiday_calendar,
                country='vnd'
            ).bulid_reset_schedule()
            fixed_rate = None
            # CurveNode.CURVENODE_CACHE.clear()
            rho_param = float(calc_rho(CURVE_NAMES=[f'{DISC_NAME}_shocked_{shock}', f'{ref_curve_name}_shocked_{shock}']).iloc[1,0])
            print(f'rho = {rho_param:.6f}')
        else:
            ref_curve = None
            fixed_rate = [float(x) for x in str(row["annual_coupon_rate"]).split(";")]
            reset_dates = []
            a_L = 0.0
            sigma_L = 0.0
            rho_param = 0.0
            ref_df = None
     
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
                fixing_lag_days=FIXING_LAG_DAYS,
                ref_curve_name=ref_curve_name,
                curve_folder=str(CURVE_FOLDER_PATH),
                ref_convention=REF_CONVENTION,
                apply_floor=apply_floor,
                apply_cap=apply_cap,
                ref_df=ref_df,
            )

        def make_tree_for_bond(sched, step_days=STEP_DAYS, **bump_kw):
            return make_tree(
                sched,
                rho_param = rho_param,
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
        straight_price = tree.decompose()['straight']

        bond_results.append({
            "bond_id": bond_id,
            "full_price": full_price,
            "straight_price": straight_price,
            "diff": full_price - straight_price,
            "shock": shock
        })

        print(f'diff = {full_price - straight_price}')
#%%

pd.DataFrame(bond_results).to_excel(
    os.path.join(root, 'outputs', 'bond_results.xlsx'),
    index=False,
)
#%%