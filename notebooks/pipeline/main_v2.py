#%%
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, date

from pathlib import Path
sys.path.insert(0, str(Path.cwd().parents[1]))

from src.bond_schedule import CouponSchedule, load_holiday_calendar
from src.bond_pricer import (
    BondTermSheet, ModelParams, PricingConfig, price_bond_layered,
)
from src.tier2_spread import (
    delta_for_bond, load_spread_table, parse_tier2_flag, validate_term_sheet,
)
from src.shock import ShockScenario
from src.calc_rho import calc_rho
from src.create_buffer_yield import calc_zyc_df

from quantmr.model.shortrate.hullwhite import HullWhite
from quantmr.curve.curvenode import CurveNode

#%%
root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH    = os.path.join(root, 'datasets', 'raw')
CURVE_FOLDER_PATH   = os.path.join(root, 'datasets', 'curve')
HOLIDAY_FOLDER_PATH = Path.cwd().parents[1] / "datasets" / "holidays"
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

VALUE_DATE = pd.to_datetime('2026-03-31')

# Nguồn DUY NHẤT của (a, sigma) cho mọi phân nhóm tổ chức phát hành.
#
# Đường ZYC của từng nhóm được dựng bằng "đường cơ bản + margin", mà margin thì
# cập nhật không đều. Hiệu chỉnh Hull-White trực tiếp trên đường nhóm vì vậy cho
# (a, sigma) không tin cậy — nó bắt cả nhiễu của margin. Mô hình lấy (a, sigma)
# hiệu chỉnh trên đường cơ bản, còn phần MỨC lãi suất vẫn khớp vào đúng đường
# nhóm qua bước quy nạp tiến Arrow-Debreu.
#
# Nói cách khác: động học vay của đường cơ bản, mức lấy của đường nhóm.
#
# Hằng số này phải là chỗ duy nhất quyết định điều đó. Trước đây hành vi này chỉ
# đúng nhờ specs/hullwhite.json tình cờ bị ghi đè cùng một bộ số cho mọi khoá —
# không có dòng code nào giữ, và một lần hiệu chỉnh lại theo tên nhóm là mất.
BASE_CURVE = 'FI_ZYC_VND_VBMA_Bond_FI'
#"act360", "act365", "actactisda"
DISC_CONVENTION = 'ACT/365'
REF_CONVENTION  = 'ACT/365'

NORM_STEP_DAYS  = float(21)
AME_STEP_DAYS   = float(5)
MIN_STEP_DAYS   = float(3)
# Giả định: ngày fixing = ngày đặt lại lãi suất (không có độ trễ).
# Ngày reset vốn đã trùng ngày trả lãi nên fixing không còn sinh mốc neo riêng
# trên lưới ngày: bỏ được 4 mốc và toàn bộ khoảng vụn 7 ngày cạnh mốc coupon.
FIXING_LAG_DAYS = float(0)

#%%
# Việc dựng lịch dòng tiền và dựng cây đã chuyển sang src/bond_pricer.py, nơi nó
# là hàm thuần theo ngày định giá. Nhờ vậy module hiệu chỉnh spread Tier 2 định
# giá được tại ngày quan sát giá thay vì chỉ tại VALUE_DATE.
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
    p = os.path.join(SPREAD_FOLDER_PATH, ten)
    if os.path.exists(p):
        return load_spread_table(p)
    print(f"[!] chưa có {p} — {nhan}")
    return None

def normalize_coupon_change_date(x):
    if pd.isna(x):
        return x

    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.strftime("%m/%d/%Y")

    if isinstance(x, str):
        return x

    return str(x)

#%%
bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'bond placeholder.csv'))
validate_term_sheet(bond_df)
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
# Đường cơ bản: bootstrap nếu chưa có, hiệu chỉnh Hull-White nếu chưa có tham số.
# Cả hai đều CÓ ĐIỀU KIỆN. Trước đây `hw.calibrate(..., save=True)` chạy vô điều
# kiện mỗi lần, tức ghi lại specs/hullwhite.json mỗi lần chạy — nên một bảng
# spread dựng trước đó lặng lẽ lệch pha với bộ tham số đang có trên đĩa.
date_columns = ["issue_date", "maturity_date", "call_exercise_dates", "put_exercise_dates", "coupon_change_date", "margin_date", "coupon_type_change_date"]
for col in date_columns:
    if col in bond_df.columns:
        bond_df[col] = bond_df[col].apply(normalize_coupon_change_date)
#%%
GROUPS = ['LB_G1', 'LB_G2', 'LB_G3', 'LB_G4', 'NBFI', 'FB', 'vbma_bond_fi']
for group in GROUPS:
    _csv = os.path.join(CURVE_FOLDER_PATH, f"FI_ZYC_VND_{group}.csv")
    ytm_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f"{group}.csv"), index_col=0, parse_dates=True)
    if not os.path.exists(_csv):
        zyc_df = calc_zyc_df(ytm_df=ytm_df).to_csv(_csv)



#%%
_base_csv = os.path.join(CURVE_FOLDER_PATH, f"{BASE_CURVE}.csv")
if not os.path.exists(_base_csv):
    calc_zyc_df(ytm_df=vbma_bond_fi).to_csv(_base_csv)

with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
    _hw_all = json.load(f)
if not {"a", "sigma"} <= set(_hw_all.get(BASE_CURVE.lower(), {})):
    print(f"Chưa có (a, sigma) cho {BASE_CURVE} — đang hiệu chỉnh Hull-White...")
    HullWhite.get(BASE_CURVE).calibrate(BASE_CURVE, method="kfmh", save=True)
    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        _hw_all = json.load(f)

A_R = _hw_all[BASE_CURVE.lower()]["a"]
SIGMA_R = _hw_all[BASE_CURVE.lower()]["sigma"]
print(f"(a, sigma) dùng cho MỌI phân nhóm, lấy từ {BASE_CURVE}: "
      f"a = {A_R:.6f}, sigma = {SIGMA_R:.6f}")
#%%

shocks_list = ["0","1", "2", "3", "4", "5", "6"]
bond_results = []

# Hai bảng phần bù, đọc một lần ngoài vòng lặp. Cả hai do build_spreads.py dựng.
# Thiếu bảng nào thì tầng đó coi như 0 — nói to chứ không im lặng.
SPREAD_FOLDER_PATH = os.path.join(root, 'datasets', 'spread')


oas_df = _doc_bang('nontier2_oas_monthly.csv',
                   'MỌI trái phiếu sẽ định giá với OAS = 0')
spread_df = _doc_bang('tier2_spread_monthly.csv',
                      'trái phiếu tăng vốn sẽ dùng delta = 0')

# Mặc định chạy cả sổ. Thu hẹp khi cần gỡ lỗi — mã American dùng step 5 ngày
# nên một mình nó chiếm phần lớn thời gian của cả lần chạy:
#   CP_ROWS=0,5,41       chạy vài dòng
#   CP_DUMP=<đường dẫn>  ghi kết quả dạng hex float để so từng bit
_rows_env = os.environ.get("CP_ROWS", "all")
ROW_SELECTION = (list(range(len(bond_df))) if _rows_env == "all"
                 else [int(x) for x in _rows_env.split(",")])

#chỉ lấy những bond có ngày phát hành < value date< ngày đáo hạn
# chuyển issue_date và maturity_date sang datetime để so sánh
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
    
    # Đường chiết khấu LUÔN là đường của phân nhóm tổ chức phát hành — kể cả với
    # trái phiếu tăng vốn. Phần bù tăng vốn vào sau, dưới dạng disc_delta.
    bond_group = spec.group
    zyc_name = f'FI_ZYC_VND_{bond_group}'
    zyc_path = Path(CURVE_FOLDER_PATH) / f"{zyc_name}.csv"

    if zyc_path.exists():
        zyc_df = pd.read_csv(zyc_path, index_col=0, parse_dates=True)
    else:
        ytm_df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{bond_group}.csv'),
                             index_col=0, parse_dates=True)
        zyc_df = calc_zyc_df(ytm_df)
        zyc_df.to_csv(zyc_path)     # chỉ ghi khi vừa bootstrap, không ghi đè mỗi vòng

    CurveNode.get(
        zyc_name,
        refresh=True,
    )

    # Hai lượng dịch trạng thái x(t) — KHÔNG phải spread cộng thẳng vào lãi suất
    # zero. build_legs tự dựng vector bump delta*B_a(tau)/tau.
    #
    #   Không tăng vốn   full = OAS          straight = 0
    #   Tăng vốn         full = OAS + delta  straight = delta
    #
    # Chân straight nằm trên đường nhóm (cộng delta nếu tăng vốn) vì đường nhóm
    # vốn dựng từ trái phiếu KHÔNG quyền chọn — nó đúng là đường cho trái phiếu
    # không quyền chọn. Hàng dưới đúng bằng "cây tăng vốn có quyền chọn trừ OAS".
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
    # Không hiệu chỉnh Hull-White theo tên nhóm: (a, sigma) lấy từ BASE_CURVE,
    # xem giải thích ở đầu file. Đường cong `zyc_df` của nhóm vẫn là đường chiết
    # khấu, và bước quy nạp tiến Arrow-Debreu vẫn khớp MỨC lãi suất vào nó.
    a_r, sigma_r = A_R, SIGMA_R


    if ref_curve_name is not None:
        
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

        # reset_dates / fixed_rate nay nằm trong BondTermSheet.
        # rho đo giữa đường cơ bản và đường tham chiếu — đúng cặp nhân tố đang
        # được mô phỏng, vì động học của chân chiết khấu lấy từ BASE_CURVE.
        rho_param = float(calc_rho(CURVE_NAMES=[BASE_CURVE, ref_curve_name]).iloc[1, 0])
        print(f'rho = {rho_param:.6f}')
    else:
        a_L = 0.0
        sigma_L = 0.0
        rho_param = 0.0
        ref_df_raw = None
    print("\n" + "-" * 50)
    print(f"{'Scenario':<15} | {'Diff':>20}")
    print("-" * 50)
    # Bản gốc của sigma, chụp TRƯỚC vòng lặp sốc. Trước đây `sigma_r = 1.25 * sigma_r`
    # ghi đè chính nó nên hệ số dồn thành 1.25**k: kịch bản 6 dùng 3.81 lần sigma gốc.
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
        # Hai chân nằm trên hai đường chiết khấu khác nhau khi OAS != 0.
        # Khi chưa có bảng OAS, hai delta bằng nhau và hàm đi nhánh một cây,
        # cho ra đúng hai con số của price_bond cũ.
        res = price_bond_layered(
            spec, VALUE_DATE,
            disc_df=disc_df, ref_df=ref_df, params=params,
            delta_full=delta_full, delta_straight=delta_straight,
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

pd.DataFrame(bond_results).to_excel(
    os.path.join(root, 'outputs', 'bond_results_tier2_oas.xlsx'),
    index=False,
)

# Bản kết xuất để so hồi quy. float.hex() là biểu diễn bit-chính xác nên hai lần
# chạy so được bằng `==` trên chuỗi, không cần dung sai.
if os.environ.get("CP_DUMP"):
    import json
    json.dump(
        [{k: (v.hex() if isinstance(v, float) else v) for k, v in r.items()}
         for r in bond_results],
        open(os.environ["CP_DUMP"], "w"), indent=1, sort_keys=True,
    )
    print(f"\n[dump] {len(bond_results)} dòng -> {os.environ['CP_DUMP']}")
        
#%%

# {
#     "sob4": {
#         "a": 0.8304433648620897,
#         "sigma": 0.03306253921457648,
#         "sigma_eps": 0.004898657265031288
#     },
#     "fi_zyc_vnd_lb_g1": {
#         "a": 0.18273340792314074,
#         "sigma": 0.017431052919058356,
#         "sigma_eps": 0.00402804405685867
#     },
#     "fi_zyc_vnd_lb_g2": {
#         "a": 0.1897463628519041,
#         "sigma": 0.021410942539161926,
#         "sigma_eps": 0.00391402745795776
#     },
#     "fi_zyc_vnd_lb_g3": {
#         "a": 1.026880483917802,
#         "sigma": 0.0220283426472668,
#         "sigma_eps": 0.007166656633431443
#     },
#     "fi_zyc_vnd_tier2_vbma": {
#         "a": 0.028014582243074265,
#         "sigma": 0.018533583408884276,
#         "sigma_eps": 0.0044531087197577915
#     },
#     "fi_zyc_vnd_tier2_vbma_bond_fi": {
#         "a": 0.30115535542342386,
#         "sigma": 0.022524596720666347,
#         "sigma_eps": 0.004204820470760032
#     },
#     "fi_zyc_vnd_vbma_bond_fi": {
#         "a": 0.3302591457703021,
#         "sigma": 0.018575040457465497,
#         "sigma_eps": 0.0036858966865396174
#     }
# }