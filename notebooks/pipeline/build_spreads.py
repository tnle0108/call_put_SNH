#%%
"""Dựng **hai** bảng phần bù từ giá dirty quan sát được, theo đúng thứ tự.

Chạy TRƯỚC ``main_v2.py``, mỗi khi một trong hai file quan sát có dòng mới::

    cd notebooks/pipeline && python build_spreads.py

Hai tầng, cộng dồn trong không gian ``x(t)``::

    Chặng 1   OAS   = phần bù của trái phiếu KHÔNG tăng vốn CÓ quyền chọn,
                      so với đường LSCK nhóm (vốn dựng từ trái phiếu không
                      quyền chọn). Nền: đường nhóm trần.
    Chặng 2   delta = phần bù thêm của trái phiếu TĂNG VỐN có quyền chọn.
                      Nền: đường nhóm ĐÃ CỘNG OAS.

Định giá dùng bốn tổ hợp::

    Không tăng vốn   full = OAS          straight = 0
    Tăng vốn         full = OAS + delta  straight = delta

Hàng dưới đúng bằng "cây tăng vốn có quyền chọn **trừ** OAS".

**Thứ tự là bắt buộc** — chặng 2 lấy kết quả chặng 1 làm nền — nên hai chặng nằm
chung một script thay vì hai file phải nhớ chạy đúng thứ tự.

Kết xuất vào ``datasets/spread/``, mỗi tầng bốn file:

``{nontier2_oas,tier2_spread}_monthly.csv``
    Bảng chính. Giá trị là lượng dịch **biến trạng thái x(t)**, KHÔNG phải spread
    cộng thẳng vào lãi suất zero. ``main_v2.py`` đọc đúng hai file này.
``*_zero_equivalent.csv``
    Bảng phái sinh ``· B_a(tau)/tau``: phần bù thực tế trên đường cong, suy giảm
    theo kỳ hạn. Dùng cho báo cáo và đối chiếu, KHÔNG dùng để định giá.
``*_provenance.csv``
    Mỗi tháng một dòng: giá trị, ``observed`` hay ``carry:<tháng>``, **số quan
    sát** và **tổng mệnh giá** đứng sau. Hai cột sau là thứ phải đọc cùng — một
    tháng "có quan sát" dựa trên một giao dịch duy nhất không đáng tin như một
    tháng có năm giao dịch, và bảng chính không thể hiện được khác biệt đó.
``*_calibrated.csv``
    Cache theo nội dung, khoá ``(bond_id, obs_date, price_hash)``.

Phương pháp luận: ``Phuong_phap_luan_spread_trai_phieu_tang_von.html``.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parents[1]))

from src.bond_pricer import ModelParams, PricingConfig                # noqa: E402
from src.bond_schedule import CouponSchedule, load_holiday_calendar   # noqa: E402
from src.calc_rho import calc_rho                                     # noqa: E402
from src.tier2_spread import (                                        # noqa: E402
    build_tier2_spread_table, calibrate_all, common_start, delta_for_bond,
    load_price_obs, validate_term_sheet, zero_equivalent,
)

root = Path.cwd().resolve().parent.parent

BOND_FOLDER_PATH = os.path.join(root, 'datasets', 'raw')
CURVE_FOLDER_PATH = os.path.join(root, 'datasets', 'curve')
SPREAD_FOLDER_PATH = os.path.join(root, 'datasets', 'spread')
HOLIDAY_FOLDER_PATH = Path.cwd().parents[1] / "datasets" / "holidays"
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

# Tháng cuối của bảng spread. Phải bằng VALUE_DATE của main_v2.py, nếu không
# `delta_for_bond` sẽ báo thiếu tháng.
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

PRICING_CFG = PricingConfig(
    curve_folder=str(CURVE_FOLDER_PATH),
    disc_convention='ACT/365',
    ref_convention='ACT/365',
    norm_step_days=float(21),
    ame_step_days=float(5),
    min_step_days=float(3),
    fixing_lag_days=float(0),
    calendar_country='vnd',
)

# ---------------------------------------------------------------------------
# Đường cong và tham số, nạp một lần mỗi nhóm
# ---------------------------------------------------------------------------
_curve_cache: dict = {}
_params_cache: dict = {}
_ref_cache: dict = {}


def _hw_params(name):
    with open(HULLWHITE_FILE_PATH, 'r', encoding='utf-8') as f:
        p = json.load(f)
    for key in (name.lower(), name):
        if key in p and "a" in p[key] and "sigma" in p[key]:
            return p[key]["a"], p[key]["sigma"]
    raise ValueError(
        f"chưa có (a, sigma) cho '{name}' trong specs/hullwhite.json. "
        f"Chạy main_v2.py một lần để hiệu chỉnh Hull-White trước."
    )


def curve_of_group(group):
    if group not in _curve_cache:
        path = Path(CURVE_FOLDER_PATH) / f"FI_ZYC_VND_{group}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} — chạy main_v2.py một lần để bootstrap đường nhóm."
            )
        _curve_cache[group] = pd.read_csv(path, index_col=0, parse_dates=True)
    return _curve_cache[group]


def ref_curve_df(name):
    if name not in _ref_cache:
        df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{name}.csv'), index_col=0)
        df.index = pd.to_datetime(df.index)
        _ref_cache[name] = df.sort_index()
    return _ref_cache[name]


def params_of_group(group, ref_curve_name=None):
    """``ModelParams`` cho một nhóm, kèm chân tham chiếu nếu mã thả nổi.

    ``group`` KHÔNG tham gia xác định ``(a_r, sigma_r)``: cả hai lấy từ
    :data:`BASE_CURVE`.  Tham số vẫn nhận ``group`` để chữ ký khỏi đánh lừa
    người đọc rằng nhóm không liên quan gì — nhóm vẫn quyết định *đường cong*
    chiết khấu, chỉ là không quyết định *động học*.

    ``rho`` đo giữa đường cơ bản và đường tham chiếu, đúng cặp nhân tố đang được
    mô phỏng.  Khoá cache là cặp ``(group, ref_curve_name)`` để giữ chữ ký ổn
    định, dù trên thực tế kết quả chỉ phụ thuộc vế sau.
    """
    key = (group, ref_curve_name)
    if key not in _params_cache:
        a_r, sigma_r = _hw_params(BASE_CURVE)
        if ref_curve_name is None:
            _params_cache[key] = ModelParams(a_r=a_r, sigma_r=sigma_r)
        else:
            a_L, sigma_L = _hw_params(ref_curve_name)
            rho = float(calc_rho(CURVE_NAMES=[BASE_CURVE, ref_curve_name]).iloc[1, 0])
            _params_cache[key] = ModelParams(a_r=a_r, sigma_r=sigma_r, a_L=a_L,
                                             sigma_L=sigma_L, rho=rho)
    return _params_cache[key]


# ---------------------------------------------------------------------------
def _bao_cao(nhan, prov, zero_eq, a_report, nhom):
    """In bảng một dòng mỗi tháng, kèm phần bù zero tương đương tại kỳ báo cáo."""
    print(f"\n{nhan} theo tháng (bp, không gian x) — a báo cáo = {a_report:.4f} ({nhom})")
    bang = prov.assign(
        gia_tri_bp=(prov["delta"] * 1e4).round(1),
        menh_gia_ty=(prov["tong_menh_gia"] / 1e9).round(0),
    )[["gia_tri_bp", "nguon", "so_quan_sat", "menh_gia_ty"]]
    print(bang.to_string(float_format=lambda v: f"{v:,.1f}", na_rep="-"))

    m = pd.Period(VALUE_DATE, freq="M")
    show = ["3M", "1Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    print(f"\n  trên lãi suất zero tại {m} (bp): "
          + "  ".join(f"{t}={zero_eq.loc[m, t] * 1e4:,.1f}" for t in show))
    n_obs = int((prov["nguon"] == "observed").sum())
    print(f"  {n_obs}/{len(prov)} tháng có quan sát, {len(prov) - n_obs} tháng bê từ kỳ trước.")


def _chang(nhan, obs, bond_df, csd, hol, *, tien_to, since, base_delta_of=None):
    """Một chặng hiệu chỉnh: dò -> gộp tháng -> ghi bốn file. Trả bảng chính."""
    calib = calibrate_all(
        obs, bond_df, csd, hol, PRICING_CFG,
        curve_of_group=curve_of_group,
        params_of=lambda spec: params_of_group(spec.group, spec.ref_curve_name),
        ref_df_of=lambda spec: (None if spec.ref_curve_name is None
                                else ref_curve_df(spec.ref_curve_name)),
        base_delta_of=base_delta_of,
        cache_path=os.path.join(SPREAD_FOLDER_PATH, f'{tien_to}_calibrated.csv'),
    )
    bad = calib[calib["status"] != "ok"]
    if len(bad):
        print(f"\n[!] {len(bad)} quan sát KHÔNG dò được, đã loại khỏi bình quân:")
        print(bad[["bond_id", "obs_date", "status"]].to_string(index=False))

    ok = calib[calib["status"] == "ok"]
    if not len(ok):
        raise ValueError(f"{nhan}: không quan sát nào dò được")

    # Quan sát trước T1 vẫn nằm trong cache để đối chiếu, nhưng KHÔNG vào bảng.
    # Với chặng 2, delta của chúng dò trên nền đường nhóm trần (tháng đó chưa có
    # OAS) nên là một đại lượng KHÁC — đừng so với delta từ T1 trở đi.
    ngoai = int((pd.to_datetime(ok["obs_date"]) < pd.Timestamp(since)).sum())
    if ngoai:
        print(f"  {ngoai}/{len(ok)} quan sát nằm trước T1 — giữ trong cache, "
              f"không đưa vào bảng.")

    nhom = ok["group"].mode()[0]
    a_report = params_of_group(nhom).a_r
    spread, zero_eq, prov = build_tier2_spread_table(
        calib, until=VALUE_DATE, since=since, a=a_report,
        out_path=os.path.join(SPREAD_FOLDER_PATH, f'{tien_to}_monthly.csv'),
        zero_path=os.path.join(SPREAD_FOLDER_PATH, f'{tien_to}_zero_equivalent.csv'),
        prov_path=os.path.join(SPREAD_FOLDER_PATH, f'{tien_to}_provenance.csv'),
    )
    _bao_cao(nhan, prov, zero_eq, a_report, nhom)
    return spread


def main():
    bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'bond placeholder.csv'))
    validate_term_sheet(bond_df)

    obs_nt2 = load_price_obs(
        os.path.join(BOND_FOLDER_PATH, 'nontier2_price_obs.csv'), bond_df,
        expect_tier2=False)
    obs_t2 = load_price_obs(
        os.path.join(BOND_FOLDER_PATH, 'tier2_price_obs.csv'), bond_df,
        expect_tier2=True)

    if not len(obs_t2):
        print("[!] tier2_price_obs.csv chưa có dòng nào — không dựng được bảng.")
        return 1
    print(f"quan sát: {len(obs_nt2)} không tăng vốn, {len(obs_t2)} tăng vốn "
          f"| ngày định giá {VALUE_DATE.date()}")

    hol = load_holiday_calendar(HOLIDAY_FOLDER_PATH)
    csd = CouponSchedule(df=bond_df, holiday_calendar=hol,
                         country='vnd').build_coupon_schedule_df()
    os.makedirs(SPREAD_FOLDER_PATH, exist_ok=True)

    if not len(obs_nt2):
        # Chưa có tầng nền: chạy đúng như trước — delta dò thẳng trên đường nhóm,
        # OAS coi như 0. Nói to chứ không im lặng, vì ý nghĩa của delta khác hẳn.
        print("\n[!] nontier2_price_obs.csv chưa có dòng nào.")
        print("    Bỏ qua chặng OAS; delta tăng vốn dò trên đường nhóm trần như")
        print("    trước. Nạp giá trái phiếu không tăng vốn có quyền chọn để bật")
        print("    kiến trúc xếp tầng.")
        since = pd.to_datetime(obs_t2["obs_date"]).min()
        _chang("delta tăng vốn", obs_t2, bond_df, csd, hol,
               tien_to='tier2_spread', since=since)
        return 0

    # T1 — mốc chung, tính lại mỗi lần chạy từ dữ liệu và VALUE_DATE.
    T1_day = common_start(obs_nt2, obs_t2, VALUE_DATE)
    T1 = pd.Timestamp(T1_day.year, T1_day.month, 1)
    print(f'T1 = {T1.date()} (tháng {pd.Period(T1, freq="M")}) — lần gần nhất cả hai tầng đều có quan sát')
    trong = lambda o: int((pd.to_datetime(o["obs_date"]).between(  # noqa: E731
        T1, VALUE_DATE)).sum())
    print(f"T1 = {T1.date()} (tháng {pd.Period(T1, freq='M')}) — lần gần nhất cả "
          f"hai tầng đều có quan sát")
    print(f"  trong cửa sổ T1..{VALUE_DATE.date()}: "
          f"{trong(obs_nt2)} quan sát không tăng vốn, {trong(obs_t2)} tăng vốn")

    print("\n" + "=" * 70 + "\nCHẶNG 1 — OAS trái phiếu không tăng vốn có quyền chọn\n" + "=" * 70)
    oas = _chang("OAS", obs_nt2, bond_df, csd, hol,
                 tien_to='nontier2_oas', since=T1)

    def nen(spec, obs_date):
        """Nền cho một quan sát tăng vốn: OAS của tháng quan sát, 0 nếu trước T1."""
        if pd.Period(pd.Timestamp(obs_date), freq="M") not in oas.index:
            return 0.0
        return delta_for_bond(oas, obs_date, spec.maturity_date)

    print("\n" + "=" * 70 + "\nCHẶNG 2 — delta trái phiếu tăng vốn, trên nền OAS\n" + "=" * 70)
    _chang("delta tăng vốn", obs_t2, bond_df, csd, hol,
           tien_to='tier2_spread', since=T1, base_delta_of=nen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#%%