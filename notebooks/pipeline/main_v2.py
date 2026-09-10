#%%
import pandas as pd
import numpy as np
import sys
import os
import json

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
from src.create_buffer_yield import BufferYTM, calc_zyc_df

from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.curve.curvenode import CurveNode
from Quant_Lib.curves import BenchmarkCurve

#%%
root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH    = os.path.join(root, 'datasets', 'raw')
CURVE_FOLDER_PATH   = os.path.join(root, 'datasets', 'curve')
HOLIDAY_FOLDER_PATH = Path.cwd().parents[1] / "datasets" / "holidays"
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

VALUE_DATE = pd.to_datetime('2026-03-31')
TIER_2_TYPE = "VBMA_Bond_FI" #VBMA, VBMA_Bond _Fi
#"act360", "act365", "actactisda"
DISC_CONVENTION = 'ACT/365'
REF_CONVENTION  = 'ACT/365'

DISC_NAME   = 'vbma_bond_fi'
DISC_TYPE = 'both vbma and vbma bond fi'  #value_date, issue_date
MARGIN_TYPE = 'effective' #effective, normal

NORM_STEP_DAYS  = float(21)
AME_STEP_DAYS   = float(5)
MIN_STEP_DAYS   = float(3)
FIXING_LAG_DAYS = float(7)

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

def make_tree(sched, rho_param, step_days=NORM_STEP_DAYS, min_step=MIN_STEP_DAYS, **kw):
    bond = compile_bond(sched, step_days=step_days, min_step=min_step)
    disc, ref = legs(**kw)
    if ref is None:
        return bond, CallPutTree.single_curve(bond, disc)
    return bond, CallPutTree.multi_curve(bond, disc, ref, rho=rho_param)

#%%
bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'bond placeholder.csv'))
vbma_bond_fi = pd.read_csv(
    os.path.join(CURVE_FOLDER_PATH, "vbma_bond_fi.csv"),
    index_col="Date",
    parse_dates=True,
)

holiday_calendar = load_holiday_calendar(HOLIDAY_FOLDER_PATH)
coupon_schedule_df = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country = 'vnd'
).build_coupon_schedule_df()

#%%

shocks_list = ["0","1", "2", "3", "4", "5", "6"]
bond_results = []

for _, row in bond_df.iloc[[41]].iterrows():
    bond_id = str(row['bond_id'])
    print("\n" + "=" * 80)
    print(bond_id)
    issue_date = pd.to_datetime(row["issue_date"])
    style = str(row["style"]).strip().lower()
    # VALUE_DATE = max(pd.to_datetime(row["issue_date"]), pd.to_datetime(vbma_bond_fi.index.min()))
    # VALUE_DATE = pd.to_datetime(row["issue_date"])
    ref_curve_name = (None if pd.isna(row['ref_curve']) else str(row['ref_curve']).strip().lower())
    maturity_date = adjust_following(pd.to_datetime(row['maturity_date']), holiday_calendar)
    coupon_accrual = float(row['coupon_accrual'])
    face_value = float(row['face'])

    if style == "american" and pd.notna(row["call_strike"]) and float(row["call_strike"]) > 0:
        call_dates = pd.date_range( start=max(issue_date,VALUE_DATE), end=maturity_date, freq=f"{str(AME_STEP_DAYS)}D" ).tolist()
        call_strikes = [ float(row["call_strike"]) / face_value ] * len(call_dates)

    elif pd.notna(row["call_exercise_dates"]) and str(row["call_exercise_dates"]).strip():
        call_dates = [pd.to_datetime(x.strip(), format='mixed') for x in str(row["call_exercise_dates"]).split(";")]
        call_strikes = [float(x)/face_value for x in str(row["call_strike"]).split(";")]
    else:
        call_dates = []
        call_strikes = []

    if style == "american" and pd.notna(row["put_strike"]) and float(row["put_strike"]) > 0:
        put_dates = pd.date_range( start=max(issue_date,VALUE_DATE), end=maturity_date, freq=f"{str(AME_STEP_DAYS)}D" ).tolist()
        put_strikes = [ float(row["put_strike"]) / face_value ] * len(put_dates)
    elif pd.notna(row["put_exercise_dates"]) and str(row["put_exercise_dates"]).strip():
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

    if style == 'american':
        step_days = AME_STEP_DAYS
    else:
        step_days = NORM_STEP_DAYS
    
    bond_group = str(row['group'])
    if bond_group == "Tier2":
        ytm_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'Tier2_{TIER_2_TYPE}.csv'), index_col=0, parse_dates=True)
        zyc_name = f'FI_ZYC_VND_{bond_group}_{TIER_2_TYPE}'
    else:
        ytm_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{bond_group}.csv'), index_col=0, parse_dates=True)
        zyc_name = f'FI_ZYC_VND_{bond_group}'
    zyc_path = Path(CURVE_FOLDER_PATH) / f"{zyc_name}.csv"

    if zyc_path.exists():
        zyc_df = pd.read_csv(zyc_path, index_col=0, parse_dates=True)
    else:          
        zyc_df = calc_zyc_df(bond=row, ytm_df=ytm_df, tier_2_type=TIER_2_TYPE)

    zyc_df.to_csv(zyc_path)
    CurveNode.get(
        zyc_name,
        refresh=True,
    )

    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        hw_params = json.load(f)
    zyc_params = hw_params.get(zyc_name.lower(), {})
    if "a" in zyc_params and "sigma" in zyc_params:
        print(f"Hull-White parameters already exist for {zyc_name}.")
        print(f"a     = {zyc_params['a']}")
        print(f"sigma = {zyc_params['sigma']}")
    else:
        print(f"Hull-White parameters not found for {zyc_name}.")
        print("Running Hull-White calibration...")
        hw = HullWhite.get(zyc_name)
        result = hw.calibrate(zyc_name, method="kfmh",save=True)
        with open(HULLWHITE_FILE_PATH, "r", encoding="utf-8") as f:
            hw_params = json.load(f)
        zyc_params = hw_params[zyc_name.lower()]

    a_r = zyc_params["a"]
    sigma_r = zyc_params["sigma"]


    if ref_curve_name is not None:
        
        # ref_ytm_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{ref_curve_name}.csv'), index_col=0, parse_dates=True)
        # ref_zyc_name = f'FI_{ref_curve_name}'
        # ref_zyc_path = Path(CURVE_FOLDER_PATH) / f"{ref_zyc_name}.csv"

        # if ref_zyc_path.exists():
        #     ref_df_raw = pd.read_csv(ref_zyc_path, index_col=0, parse_dates=True)
        # else:          
        #     ref_df_raw = calc_zyc_df(bond=row, ytm_df=ytm_df)


        ref_df_raw = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{ref_curve_name}.csv'), index_col = 0)

        ref_df_raw.index = pd.to_datetime(ref_df_raw.index)
        ref_df_raw = ref_df_raw.sort_index()

        with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
            hw_params = json.load(f)

        ref_curve_params = hw_params.get(ref_curve_name, {})

        if "a" in ref_curve_params and "sigma" in ref_curve_params:
            print(f"Hull-White parameters already exist for {ref_curve_name}.")
            print(f"a     = {ref_curve_params['a']}")
            print(f"sigma = {ref_curve_params['sigma']}")
        else:
            print(f"Hull-White parameters not found for {ref_curve_name}.")
            print("Running Hull-White calibration...")

            hw = HullWhite.get(ref_curve_name)
            result = hw.calibrate(ref_curve_name, method="kfmh", save=True)
            with open(HULLWHITE_FILE_PATH, "r", encoding="utf-8") as f:
                hw_params = json.load(f)
            ref_curve_params = hw_params[ref_curve_name]

        a_L = ref_curve_params["a"]
        sigma_L = ref_curve_params["sigma"]

        reset_dates = CouponSchedule(
            df=bond_df[bond_df['bond_id'] == row['bond_id']],
            holiday_calendar=holiday_calendar,
            country='vnd'
        ).bulid_reset_schedule()
        fixed_rate = None
        rho_param = float(calc_rho(CURVE_NAMES=[f'{zyc_name}', f'{ref_curve_name}']).iloc[1,0])
        print(f'rho = {rho_param:.6f}')
    else:
        fixed_rate = [float(x) for x in str(row["annual_coupon_rate"]).split(";")]
        reset_dates = []
        a_L = 0.0
        sigma_L = 0.0
        rho_param = 0.0
        ref_df_raw = None
    print("\n" + "-" * 50)
    print(f"{'Scenario':<15} | {'Diff':>20}")
    print("-" * 50)
    for shock in shocks_list:
        if shock == "0":
            disc_df = zyc_df
            ref_df = ref_df_raw
            sigma_L = sigma_L
            sigma_r = sigma_r
        else:
            disc_df = ShockScenario(shock_type=shock, df = zyc_df).create_shock_df()
            ref_df = (ShockScenario(shock_type=shock, df = ref_df_raw).create_shock_df() if ref_df_raw is not None else None)
            sigma_r = 1.25 * sigma_r
            sigma_L = 1.25 * sigma_L

        disc_curve = MapCurve(rpd=VALUE_DATE, df = disc_df, convention=DISC_CONVENTION).map_curve()
        ref_curve = (MapCurve(rpd=VALUE_DATE, df=ref_df, convention=REF_CONVENTION).map_curve() if ref_df is not None else None)

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
        def make_tree_for_bond(sched, step_days=step_days, min_step=MIN_STEP_DAYS, **bump_kw):
            return make_tree(
                sched,
                rho_param = rho_param,
                step_days=step_days,
                min_step=min_step,
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
        diff = full_price - straight_price
        bond_results.append({
            "bond_id": bond_id,
            "full_price": full_price,
            "straight_price": straight_price,
            "diff": diff,
            "shock": shock
        })

        print(f"{f'SHOCK_{shock}':<15} | {diff:>20.6f}")
    print("-" * 50)
#%%

pd.DataFrame(bond_results).to_excel(
    os.path.join(root, 'outputs', f'bond_results_with_Tier_2_type_{TIER_2_TYPE}.xlsx'),
    index=False,
)
        
#%%
