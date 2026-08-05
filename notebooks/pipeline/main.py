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

#%%
root = Path.cwd().resolve().parent.parent

BOND_DATA_PATH = root / "datasets" / "raw" / "bonds_placeholder.csv"
CURVE_DATA_PATH = root / "datasets" / "raw" / "Histocopy_FI_ZYC_VND_GD2_family.xlsx"
RAW_PATH = root/'datasets'/'raw'

START_DATE = pd.to_datetime("2024-06-24")
END_DATE = pd.to_datetime("2026-07-30")
REPORT_DATE = pd.to_datetime("2026-06-03")

DISC_CONVENTION = "actactisda"
COUP_CONVENTION = "act365"
#%%


bond_df = pd.read_csv(BOND_DATA_PATH)
bond_df["issue_date"] = pd.to_datetime(bond_df["issue_date"])
bond_df["maturity_date"] = pd.to_datetime(bond_df["maturity_date"])
# bond_df['exercise_dates'] = bond_df['exercise_dates'].fillna("").astype(str).str.split(";")

# exercise_map = bond_df[['bond_id, exercise_dates']].copy()
# # exercise_map = pd.read_csv(EXERCISE_MAP_PATH)
# # exercise_map["bond_id"] = exercise_map["bond_id"].astype(str)
# # exercise_map["exercise_dates"] = exercise_map["exercise_dates"].fillna("").astype(str)
# # exercise_map["exercise_dates"] = exercise_map["exercise_dates"].str.split(";")

# # Convert mapping to dict {bond_id: [exercise_dates]}
# exercise_dates_by_bond = {
#     row["bond_id"]: [pd.to_datetime(d) for d in row["exercise_dates"] if str(d).strip() != ""]
#     for _, row in exercise_map.iterrows()
# }

def normalize_date_list(value):
    if pd.isna(value) or str(value).strip() == "":
        return value

    dates = [
        pd.to_datetime(x.strip(), format="mixed").strftime("%Y-%m-%d")
        for x in str(value).split(";")
    ]
    return ";".join(dates)

bond_df["call_exercise_dates"] = bond_df["call_exercise_dates"].apply(normalize_date_list)
bond_df["put_exercise_dates"] = bond_df["put_exercise_dates"].apply(normalize_date_list)

bond_df["exercise_dates"] = (
    bond_df["call_exercise_dates"].fillna("").astype(str)
    + ";"
    + bond_df["put_exercise_dates"].fillna("").astype(str)
).str.strip(";")

exercise_dates_by_bond = {
    row["bond_id"]: [
        pd.to_datetime(d.strip())
        for d in str(row["exercise_dates"]).split(";")
        if d.strip()
    ]
    for _, row in bond_df.iterrows()
}

# Fallback if exercise date file is not present
def get_exercise_dates_for_bond(bond_id: str):
    if bond_id in exercise_dates_by_bond:
        return exercise_dates_by_bond[bond_id]
    return []
#%%
# ============================================================
# 3) Build per-group curve dictionary
# ============================================================
groups = ["LB_G1", "LB_G2", "LB_G3", "LB_G4", "NBFI", "FB"]

dfs = {}
curves = {}
ref_curves = {}

daycount = DayCount.get(DISC_CONVENTION)

def map_disc_curve(group):
    df = pd.read_excel(
        CURVE_DATA_PATH,
        sheet_name=f"FI ZYC VND_GD2_{group}",
        index_col=0,
    )
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df.reindex(pd.date_range(start=START_DATE, end=END_DATE))
    dfs[group] = df

    # infer maturities from tenor labels, e.g. 3M, 6M, 1Y, 2Y, ...
    tenor_labels = [col.split("_")[-1] for col in df.columns]
    maturity_dates = []

    for tenor in tenor_labels:
        if tenor.endswith("Y"):
            n = int(tenor[:-1])
            maturity_dates.append(REPORT_DATE + pd.DateOffset(years=n))
        elif tenor.endswith("M"):
            n = int(tenor[:-1])
            maturity_dates.append(REPORT_DATE + pd.DateOffset(months=n))
        else:
            raise ValueError(f"Unsupported tenor: {tenor}")

    start = np.array([REPORT_DATE] * len(maturity_dates), dtype="datetime64[D]")
    end = np.array(maturity_dates, dtype="datetime64[D]")
    maturities = daycount.yearfrac(start, end)

    # pick the last valid rate point on or before REPORT_DATE
    valid_idx = df.index[df.index <= REPORT_DATE]
    report_idx = valid_idx.max()
    zero_rates = df.loc[report_idx].to_numpy(dtype=float)

    curves[group] = YieldCurve.from_zero_rates(
        maturities=maturities,
        zero_rates=zero_rates,
    )
    return curves[group]

def map_ref_curve(ref_curve_name):
    df = pd.read_csv(
        os.path.join(RAW_PATH,f'{ref_curve_name}.csv'),
        index_col=0,
)


    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    tenor_labels = [col for col in df.columns]
    maturity_dates=[]
    
    for tenor in tenor_labels:
        if tenor.endswith("Y"):
            n = int(tenor[:-1])
            maturity_dates.append(REPORT_DATE + pd.DateOffset(years=n))
        elif tenor.endswith("M"):
            n = int(tenor[:-1])
            maturity_dates.append(REPORT_DATE + pd.DateOffset(months=n))
        elif tenor.endswith("N"):
            n = 1
            maturity_dates.append(REPORT_DATE + pd.DateOffset(days=n))
        elif tenor.endswith("W"):
            n = int(tenor[:-1])
            maturity_dates.append(REPORT_DATE + pd.DateOffset(weeks=n))
        else:
            raise ValueError(f"Unsupported tenor: {tenor}")
    start = np.array([REPORT_DATE] * len(maturity_dates), dtype="datetime64[D]")
    end = np.array(maturity_dates, dtype="datetime64[D]")
    maturities = daycount.yearfrac(start, end)

    valid_idx = df.index[df.index <= REPORT_DATE]
    report_idx = valid_idx.max()
    zero_rates = df.loc[report_idx].to_numpy(dtype=float)

    curve = YieldCurve.from_zero_rates(
        maturities = maturities,
        zero_rates = zero_rates,
    )

    return curve

def infer_tree_grid_for_bond(row: pd.Series, daycount_convention: str = "act365"):
    """
    Từ issue_date, coupon_dates, exercise_dates, maturity_date:
      1) tính khoảng cách ngày giữa các event liên tiếp
      2) lấy gcd của các khoảng cách ngày
      3) đặt dt = gcd_days / 365
      4) kiểm tra các yearfraction event có rơi đúng lên node không
    """
    bond_id = str(row["bond_id"])
    issue_date = pd.to_datetime(row["issue_date"])
    maturity_date = pd.to_datetime(row["maturity_date"])

    # coupon dates theo coupon_accrual
    coupon_accrual = float(row["coupon_accrual"])
    maturity_date = pd.to_datetime(row["maturity_date"], format="mixed")

    months = int(round(coupon_accrual * 12))

    coupon_dates = list(
        pd.date_range(
            start=issue_date,
            end=maturity_date,
            freq=pd.DateOffset(months=months)
        )[1:]   # bỏ issue_date
    )

    # exercise dates from lookup
    exercise_dates = get_exercise_dates_for_bond(bond_id)
    exercise_dates = sorted(set(exercise_dates))

    # tạo event list
    event_dates = sorted(set(coupon_dates + exercise_dates + [maturity_date]))

    # 1) khoảng cách ngày giữa các event liên tiếp
    day_gaps = np.diff(
        np.array([issue_date] + event_dates, dtype="datetime64[D]")
    ).astype(int)

    # 2) gcd của các khoảng cách ngày
    x_days = int(day_gaps[0])
    for d in day_gaps[1:]:
        x_days = gcd(x_days, int(d))

    # 3) dt theo năm
    dt = x_days / 365.0

    # 4) year fraction theo daycount thực tế
    daycount_obj = DayCount.get(daycount_convention)
    times = daycount_obj.yearfrac(
        np.array([issue_date] * len(event_dates), dtype="datetime64[D]"),
        np.array(event_dates, dtype="datetime64[D]"),
    )
    t_sorted = np.sort(times)

    # 5) ép các time về grid
    grid_times = np.round(t_sorted / dt) * dt

    # 6) nếu lệch thì báo lỗi
    if not np.allclose(t_sorted, grid_times, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"Bond {bond_id} cannot be aligned to a common dt={dt} grid "
            f"with gcd-days={x_days}."
        )

    # n_steps là số bước tới maturity trên grid đó
    n_steps = int(round(t_sorted[-1] / dt))

    return dt, n_steps

def tenor_to_yearfrac(tenor, convention, date):

    maturity_dates=[]
    if tenor.endswith("Y"):
        n = int(tenor[:-1])
        maturity_dates.append(REPORT_DATE + pd.DateOffset(years=n))
    elif tenor.endswith("M"):
        n = int(tenor[:-1])
        maturity_dates.append(REPORT_DATE + pd.DateOffset(months=n))
    elif tenor.endswith("N"):
        n = 1
        maturity_dates.append(REPORT_DATE + pd.DateOffset(days=n))
    elif tenor.endswith("W"):
        n = int(tenor[:-1])
        maturity_dates.append(REPORT_DATE + pd.DateOffset(weeks=n))
    else:
        raise ValueError(f"Unsupported tenor: {tenor}")
    start = np.array([date] * len(maturity_dates), dtype="datetime64[D]")
    end = np.array(maturity_dates, dtype="datetime64[D]")

    yf = DayCount.get(convention).yearfrac(start, end)
    return yf


def build_coupon_schedule_df(bond_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, bond in bond_df.iterrows():
        issue_date = pd.to_datetime(bond["issue_date"])
        maturity_date = pd.to_datetime(bond["maturity_date"])

        accrual = float(bond["coupon_accrual"])
        months = int(round(accrual * 12))

        pay_dates = pd.date_range(
            start=issue_date,
            end=maturity_date,
            freq=pd.DateOffset(months=months),
        )[1:]       

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


def build_coupon_map_from_schedule(row: pd.Series, coupon_schedule_df: pd.DataFrame):
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

def build_exercise_spec(
    bond_row: pd.Series,
    date_to_step: dict[pd.Timestamp, int],
) -> ExerciseSpec:

    call = {}
    put = {}

    if pd.notna(bond_row["call_exercise_dates"]):
        dates = [pd.to_datetime(x.strip()) for x in str(bond_row["call_exercise_dates"]).split(";")]
        strikes = [float(x) for x in str(bond_row["call_strike"]).split(";")]

        call = {date_to_step[d]: s for d, s in zip(dates, strikes)}

    if pd.notna(bond_row["put_exercise_dates"]):
        dates = [pd.to_datetime(x.strip()) for x in str(bond_row["put_exercise_dates"]).split(";")]

        strikes = [float(x) for x in str(bond_row["put_strike"]).split(";")]

        put = {date_to_step[d]: s for d, s in zip(dates, strikes)}

    return ExerciseSpec(
        call=call,
        put=put,
    )
    
#%%
coupon_schedule_df = build_coupon_schedule_df(bond_df)


#%%

results = []

for _, row in bond_df.iterrows():
    bond_id = str(row["bond_id"])
    issue_date = pd.to_datetime(row["issue_date"])
    ref_curve_name = (None if pd.isna(row["ref_curve"]) else str(row["ref_curve"]).strip().lower())
    print(ref_curve_name)
    style = str(row['style']).strip().lower()
    dt, n_steps = infer_tree_grid_for_bond(row)

    group = str(row["group"]).strip() if "group" in row and pd.notna(row["group"]) else None
    # if group not in curves:
    #     raise ValueError(f"Group {group} not found in curves.")

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

    def to_step(date):
        date = pd.Timestamp(date).normalize()
        t = (date - issue_date.normalize()).days / 365.0
        return int(round(t / dt))

    call = {to_step(d): s for d, s in zip(call_dates, call_strikes)}
    put = {to_step(d): s for d, s in zip(put_dates, put_strikes)}

    disc_curve = map_disc_curve(group)
    if ref_curve_name is not None:
        ref_curve = map_ref_curve(ref_curve_name)
    else:
        ref_curve = None


    tree = MultiCurveHWTree(
        a_r=0.03,
        sigma_r=0.015,
        a_L=0.02,
        sigma_L=0.018,
        rho=0.20,
        disc_curve=disc_curve,
        ref_curve=ref_curve,
        dt=dt,
        n_steps=n_steps,
    )

    maturity_step = n_steps
    face = float(row["face"])
    coupon_accrual = float(row["coupon_accrual"])
    coupon_map = build_coupon_map_from_schedule(row, coupon_schedule_df)
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
        "dt": dt,
        "n_steps": n_steps,
        "full_price": full_price,
        "straight_price": straight_price,
    })

results_df = pd.DataFrame(results)
results_df

#%%
