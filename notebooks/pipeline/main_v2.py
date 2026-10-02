#%%
import pandas as pd
import sys
import os
import json
from datetime import datetime, date

from pathlib import Path

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass
sys.path.insert(0, str(Path.cwd().parents[1]))

from src.bond_schedule import CouponSchedule, load_holiday_calendar, tenor_to_years
from src.bond_pricer import (
    BondTermSheet, ModelParams, PricingConfig, price_bond_layered,
)
from src.tier2_spread import (
    delta_for_bond, load_spread_table, parse_tier2_flag, validate_term_sheet,
)
from src.shock import ShockScenario
from src.calc_rho import calc_rho

from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.curve.curvenode import CurveNode

#%%
"""Define paths and constants for bond pricing pipeline."""
root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH    = os.path.join(root, 'datasets', 'raw') # Bond transaction data needed to be valued
CURVE_FOLDER_PATH   = os.path.join(root, 'datasets', 'curve') # ZYC data for reference curves and discount curves
HOLIDAY_FOLDER_PATH = os.path.join(root, 'datasets', 'holiday') # Holiday calendar for Vietnam
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json') # Hull-White parameters for reference curves and discount curves

VALUE_DATE = pd.to_datetime('2026-03-31')

BASE_CURVE = 'FI_ZYC_VND_VBMA_Bond_FI'

# Hull-White calibration tenors (years) per curve, as set in the Hull-White document
# (20260819_TLXDMH_HW §2.3, "Chuẩn bị"). A reference curve not listed uses its own tenors.
CALIBRATION_TENORS = {
    BASE_CURVE.lower(): [3/12, 6/12, 9/12, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
                            # 5.0, 7.0, 10.0, 15.0, 20.0, 30.0

    ],
    'sob4': [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 4.0, 5.0]
}

DISC_CONVENTION = 'ACT/365'
REF_CONVENTION  = 'ACT/365'

NORM_STEP_DAYS  = float(21)
AME_STEP_DAYS   = float(5)
MIN_STEP_DAYS   = float(3)
FIXING_LAG_DAYS = float(0) # Assumption: fixing date = rate reset date (no lag).

#%%
"""Define pricing config and create helpers"""
# Create a PricingConfig object to hold the configuration for pricing bonds.
PRICING_CFG = PricingConfig(
    curve_folder=str(CURVE_FOLDER_PATH),
    disc_convention=DISC_CONVENTION,
    ref_convention=REF_CONVENTION,
    norm_step_days=NORM_STEP_DAYS,
    ame_step_days=AME_STEP_DAYS,
    min_step_days=MIN_STEP_DAYS,
    fixing_lag_days=FIXING_LAG_DAYS,
    calendar_country='vnd',
)

def _doc_bang(ten, nhan):
    """
    Load a spread table from SPREAD_FOLDER_PATH.

    A missing table is reported loudly rather than silently, and the caller treats that layer
    as 0.

    Args:
        ten (str): The filename of the spread table to load.
        nhan (str): What happens without the table, printed in the warning.

    Returns:
        pd.DataFrame or None: The loaded spread table, or None if the file does not exist.
    """
    p = os.path.join(SPREAD_FOLDER_PATH, ten)
    if os.path.exists(p):
        return load_spread_table(p)
    print(f"[!] chưa có {p} — {nhan}")
    return None

def calibration_tenors(curve_name):
    """
    Hull-White calibration tenors of a curve, in years.

    Args:
        curve_name (str): Curve name (case-insensitive).

    Returns:
        list[float]: The tenors set in CALIBRATION_TENORS, or for a curve not listed there its own
        tenor columns (overnight and week tenors excluded, as in the calibration itself).
    """
    if curve_name.lower() in CALIBRATION_TENORS:
        return CALIBRATION_TENORS[curve_name.lower()]
    terms = CurveNode.get(curve_name).terms
    return [tenor_to_years(x) for x in terms if x not in ("ON", "1W", "2W")]

def normalize_coupon_change_date(x):
    """
    Normalize one value of a date column in the bond DataFrame.

    Args:
        x (str | datetime | NaN): The value to normalize.

    Returns:
        str or NaN: The date as a "MM/DD/YYYY" string (strings pass through unchanged), or NaN
        if the input was NaN.
    """
    if pd.isna(x):
        return x

    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.strftime("%m/%d/%Y")

    if isinstance(x, str):
        return x

    return str(x)

#%%
"""Read bond data, validate term sheet, and load VBMA bond yield curve."""
bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'term_sheet.csv')) # Read raw bond data from CSV file
validate_term_sheet(bond_df) # Validate the term sheet data in the bond DataFrame, ensure that `group` and `is_tier2` columns are correctly formatted.

holiday_calendar = load_holiday_calendar(HOLIDAY_FOLDER_PATH) # Load the holiday calendar for Vietnam from the specified folder path, which will be used for date calculations in bond pricing.
coupon_schedule_df = CouponSchedule(
    df=bond_df,
    holiday_calendar=holiday_calendar,
    country = 'vnd'
).build_coupon_schedule_df()
date_columns = ["issue_date", "maturity_date", "call_exercise_dates", "put_exercise_dates", "coupon_change_date", "margin_date", "coupon_type_change_date"]
for col in date_columns:
    if col in bond_df.columns:
        bond_df[col] = bond_df[col].apply(normalize_coupon_change_date)

#%%
# Base curve: bootstrap it if missing, calibrate Hull-White if its parameters are missing.
# Both steps are CONDITIONAL. `hw.calibrate(..., save=True)` used to run unconditionally,
# rewriting specs/hullwhite.json on every run, so a previously built spread table silently
# drifted out of sync with the parameter set on disk.
# _base_csv = os.path.join(CURVE_FOLDER_PATH, f"{BASE_CURVE}.csv")


with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
    _hw_all = json.load(f)
if not {"a", "sigma"} <= set(_hw_all.get(BASE_CURVE.lower(), {})):
    print(f"Chưa có (a, sigma) cho {BASE_CURVE} — đang hiệu chỉnh Hull-White...")
    HullWhite.get(BASE_CURVE).calibrate(
        BASE_CURVE, method="kfmh", save=True, tau=calibration_tenors(BASE_CURVE))
    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        _hw_all = json.load(f)

A_R = _hw_all[BASE_CURVE.lower()]["a"]
SIGMA_R = _hw_all[BASE_CURVE.lower()]["sigma"]
print(f"(a, sigma) dùng cho MỌI phân nhóm, lấy từ {BASE_CURVE}: "
      f"a = {A_R:.6f}, sigma = {SIGMA_R:.6f}")
#%%

shocks_list = ["0","1", "2", "3", "4", "5", "6"]
RHO_BY_REF_CURVE = {}  # rho per reference curve, filled on first use
bond_results = []

# Two spread tables, read once outside the loop. Both are built by build_spreads.py.
# A missing table means that layer is treated as 0, reported loudly rather than silently.
SPREAD_FOLDER_PATH = os.path.join(root, 'datasets', 'spread')


oas_df = _doc_bang('nontier2_oas_monthly.csv',
                   'MỌI trái phiếu sẽ định giá với OAS = 0')
spread_df = _doc_bang('tier2_spread_monthly.csv',
                      'trái phiếu tăng vốn sẽ dùng delta = 0')

# Runs the whole book by default. Narrow it when debugging: the American bond uses a 5-day
# step, so it alone takes most of the run time:
#   CP_ROWS=0,5,41       run only these rows
#   CP_DUMP=<path>       write results as hex floats for bit-exact comparison
_rows_env = os.environ.get("CP_ROWS", "all")
ROW_SELECTION = (list(range(len(bond_df))) if _rows_env == "all"
                 else [int(x) for x in _rows_env.split(",")])

# Keep only bonds with issue date <= value date < maturity date.
# Convert issue_date and maturity_date to datetime for the comparison.
bond_df["issue_date"] = pd.to_datetime(bond_df["issue_date"])
bond_df["maturity_date"] = pd.to_datetime(bond_df["maturity_date"])
ROW_SELECTION = [i for i in ROW_SELECTION if bond_df.iloc[i]["issue_date"] <= VALUE_DATE < bond_df.iloc[i]["maturity_date"]]

for i in ROW_SELECTION:
    row = bond_df.iloc[i]
    spec = BondTermSheet.from_bond_df(
        bond_df, coupon_schedule_df, holiday_calendar, PRICING_CFG, iloc=i)
    bond_id = spec.bond_id
    print("\n" + "=" * 120)
    print(bond_id)
    ref_curve_name = spec.ref_curve_name
    maturity_date = spec.maturity_date
    
    # The discount curve is ALWAYS the issuer group's curve, Tier-2 bonds included.
    # The Tier-2 spread comes in afterwards, as disc_delta.
    bond_group = spec.group
    zyc_name = f'FI_ZYC_VND_{bond_group}'
    zyc_path = Path(CURVE_FOLDER_PATH) / f"{zyc_name}.csv"
    zyc_df = pd.read_csv(zyc_path, index_col=0, parse_dates=True)

    CurveNode.get(
        zyc_name,
        refresh=True,
    )
    zyc_df = zyc_df[zyc_df.index <= VALUE_DATE]
    # Two shifts of the state variable x(t), NOT spreads added directly to zero rates.
    # build_legs builds the bump vector delta*B_a(tau)/tau itself.
    #
    #   Non-Tier-2   full = OAS          straight = 0
    #   Tier-2       full = OAS + delta  straight = delta
    #
    # The straight leg sits on the group curve (plus delta for Tier-2) because the group
    # curve is built from bonds WITHOUT options, so it is exactly the curve for an option-free
    # bond. The bottom row equals "the callable Tier-2 tree minus OAS".
    is_tier2 = parse_tier2_flag(row.get('is_tier2'))
    oas = (0.0 if oas_df is None
           else delta_for_bond(oas_df, VALUE_DATE, spec.maturity_date))
    delta_t2 = (delta_for_bond(spread_df, VALUE_DATE, spec.maturity_date)
                if is_tier2 and spread_df is not None else 0.0)
    if is_tier2 and spread_df is None:
        print("  [!] trái phiếu tăng vốn nhưng chưa có bảng spread -> delta = 0")
    delta_full, delta_straight = oas + delta_t2, delta_t2
    print(f"  OAS = {oas * 1e4:+.1f} bp | delta = {delta_t2 * 1e4:+.1f} bp"
          f" -> full {delta_full * 1e4:+.1f} / straight {delta_straight * 1e4:+.1f}")

    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        hw_params = json.load(f)
    # No Hull-White calibration per group name: (a, sigma) come from BASE_CURVE (see the
    # rationale at BASE_CURVE in build_spreads.py). The group's `zyc_df` is still the discount
    # curve, and the Arrow-Debreu forward induction still fits the rate LEVEL to it.
    a_r, sigma_r = A_R, SIGMA_R


    if ref_curve_name is not None:
        
        ref_df_raw = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{ref_curve_name}.csv'), index_col = 0)

        ref_df_raw.index = pd.to_datetime(ref_df_raw.index)
        ref_df_raw = ref_df_raw.sort_index()
        ref_df_raw = ref_df_raw[ref_df_raw.index <= VALUE_DATE]

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
            hw.calibrate(ref_curve_name, method="kfmh", save=True,
                         tau=calibration_tenors(ref_curve_name))
            with open(HULLWHITE_FILE_PATH, "r", encoding="utf-8") as f:
                hw_params = json.load(f)
            ref_curve_params = hw_params[ref_curve_name]

        a_L = ref_curve_params["a"]
        sigma_L = ref_curve_params["sigma"]

        # reset_dates / fixed_rate now live in BondTermSheet.
        # rho is measured between the base curve and the reference curve: exactly the factor
        # pair being simulated, since the discount leg's dynamics come from BASE_CURVE.
        # Same pair for every bond on this reference curve: estimate once per run (HW doc,
        # Phụ lục 07: whole history up to VALUE_DATE) and keep a copy in corr.csv.
        if ref_curve_name not in RHO_BY_REF_CURVE:
            RHO_BY_REF_CURVE[ref_curve_name] = float(calc_rho(
                CURVE_NAMES=[BASE_CURVE, ref_curve_name], value_date=VALUE_DATE, save=True,
            ).iloc[1, 0])
        rho_param = RHO_BY_REF_CURVE[ref_curve_name]
        print(f'rho = {rho_param:.6f}')
    else:
        a_L = 0.0
        sigma_L = 0.0
        rho_param = 0.0
        ref_df_raw = None
    print("\n" + "-" * 50)
    print(f"{'Scenario':<15} | {'Diff':>20}")
    print("-" * 50)
    # Original sigma, snapshotted BEFORE the shock loop. `sigma_r = 1.25 * sigma_r` used to
    # overwrite itself, so the factor compounded to 1.25**k: scenario 6 used 3.81x the base
    # sigma.
    sigma_r0, sigma_L0 = sigma_r, sigma_L
    for shock in shocks_list:
        if shock == "0":
            disc_df = zyc_df
            ref_df = ref_df_raw
            sigma_r, sigma_L = sigma_r0, sigma_L0
        else:
            disc_df = ShockScenario(shock_type=shock, df = zyc_df).create_shock_df()
            ref_df = (ShockScenario(shock_type=shock, df = ref_df_raw).create_shock_df() if ref_df_raw is not None else None)
            sigma_r, sigma_L = 1.25 * sigma_r0, 1.25 * sigma_L0

        params = ModelParams(a_r=a_r, sigma_r=sigma_r,
                             a_L=a_L, sigma_L=sigma_L, rho=rho_param)
        # The two legs sit on two different discount curves when OAS != 0.
        # Without an OAS table the two deltas are equal and the function takes the one-tree
        # branch, reproducing exactly the two numbers of the old price_bond.
        res = price_bond_layered(
            spec, VALUE_DATE,
            disc_df=disc_df, ref_df=ref_df, params=params,
            delta_full=delta_full, delta_straight=delta_straight,
            ref_df_raw=ref_df_raw
        )
        full_price = res.full_price
        straight_price = res.straight_price
        diff = res.diff
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
value_date_str = VALUE_DATE.strftime("%Y%m%d")
pd.DataFrame(bond_results).to_excel(
    os.path.join(root, 'outputs', f'{value_date_str}_bond_results_tier2_oas.xlsx'),
    index=False,
)

# Dump for regression comparison. float.hex() is a bit-exact representation, so two runs
# can be compared with `==` on the strings, with no tolerance.
if os.environ.get("CP_DUMP"):
    import json
    json.dump(
        [{k: (v.hex() if isinstance(v, float) else v) for k, v in r.items()}
         for r in bond_results],
        open(os.environ["CP_DUMP"], "w"), indent=1, sort_keys=True,
    )
    print(f"\n[dump] {len(bond_results)} dòng -> {os.environ['CP_DUMP']}")
#%%

    # "fi_zyc_vnd_vbma_bond_fi": {
    #     "a": 0.3302591457703021,
    #     "sigma": 0.018575040457465497,
    #     "sigma_eps": 0.0036858966865396174
    # }