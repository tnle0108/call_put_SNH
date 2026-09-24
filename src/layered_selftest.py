"""Phép kiểm cho :func:`src.bond_pricer.price_bond_layered`.

Chạy:  python src/layered_selftest.py        (từ thư mục gốc)

Cần dữ liệu thật (đường cong nhóm + term sheet) nên chậm hơn hai selftest kia —
vài chục giây. Nó kiểm đúng ba điều mà hai selftest hàm thuần không với tới:

1. **Cổng hồi quy.** Khi hai delta bằng nhau, `price_bond_layered` phải cho ra
   **đúng** hai con số của `price_bond`, tới từng bit. Đây là điều bảo đảm rằng
   bật kiến trúc xếp tầng lên mà chưa có OAS thì không có gì xê dịch.
2. **Nhánh hai cây chạy đúng.** Ép đi nhánh hai cây với hai delta *khác nhau một
   lượng vô cùng nhỏ* — kết quả phải gần như trùng nhánh một cây. Nếu nhánh hai
   cây dựng sai lưới hay sai chân, chênh lệch sẽ lộ ra ngay.
3. **Hai cây cùng hình học lưới.** `FactorLattice` không đọc đường cong, nên hai
   cây chỉ khác nhau ở `phi` / `step_discount`. Đây là điều khiến sai số rời rạc
   hoá vẫn triệt tiêu khi lấy hiệu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bond_pricer import (                                        # noqa: E402
    BondTermSheet, ModelParams, PricingConfig, build_tree, price_bond,
    price_bond_layered,
)
from src.bond_schedule import CouponSchedule, load_holiday_calendar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "datasets" / "curve"
VALUE_DATE = pd.Timestamp("2026-03-31")

_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


# --------------------------------------------------------------------------
bond_df = pd.read_csv(ROOT / "datasets" / "raw" / "bond placeholder.csv")
hol = load_holiday_calendar(ROOT / "datasets" / "holidays")
csd = CouponSchedule(df=bond_df, holiday_calendar=hol, country="vnd").build_coupon_schedule_df()
CFG = PricingConfig(curve_folder=str(CURVE), norm_step_days=21.0, ame_step_days=5.0,
                    min_step_days=3.0, fixing_lag_days=0.0, calendar_country="vnd")

ids = bond_df["bond_id"].astype(str).str.strip().tolist()
# Một mã lãi cố định và một mã thả nổi hai nhân tố — hai đường mã khác hẳn nhau.
MAU = [b for b in ("VCB12601", "VCB12604") if b in ids]
assert MAU, "không tìm thấy mã mẫu trong term sheet"


def _spec(bond_id):
    return BondTermSheet.from_bond_df(bond_df, csd, hol, CFG, iloc=ids.index(bond_id))


def _inputs(spec):
    zyc = pd.read_csv(CURVE / f"FI_ZYC_VND_{spec.group}.csv", index_col=0, parse_dates=True)
    if spec.ref_curve_name is None:
        return zyc, None, ModelParams(a_r=0.3303, sigma_r=0.018575)
    ref = pd.read_csv(CURVE / f"{spec.ref_curve_name}.csv", index_col=0)
    ref.index = pd.to_datetime(ref.index)
    return zyc, ref.sort_index(), ModelParams(
        a_r=0.3303, sigma_r=0.018575, a_L=0.8304, sigma_L=0.033063, rho=-0.0629)


print("\n1 · Cổng hồi quy: hai delta bằng nhau -> trùng price_bond từng bit")
for bond_id in MAU:
    spec = _spec(bond_id)
    zyc, ref, par = _inputs(spec)
    for d in (0.0, -0.0211):
        cu = price_bond(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par,
                        disc_delta=d)
        moi = price_bond_layered(spec, VALUE_DATE, disc_df=zyc, ref_df=ref,
                                 params=par, delta_full=d, delta_straight=d)
        check(f"{bond_id} delta={d:+.4f}: full trùng bit",
              cu.full_price.hex() == moi.full_price.hex())
        check(f"{bond_id} delta={d:+.4f}: straight trùng bit",
              cu.straight_price.hex() == moi.straight_price.hex())


print("\n2 · Nhánh hai cây")
EPS = 1e-12
for bond_id in MAU:
    spec = _spec(bond_id)
    zyc, ref, par = _inputs(spec)
    mot = price_bond_layered(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par,
                             delta_full=0.0, delta_straight=0.0)
    hai = price_bond_layered(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par,
                             delta_full=EPS, delta_straight=0.0)
    check(f"{bond_id}: hai delta lệch {EPS:g} -> giá gần như trùng nhánh một cây",
          abs(hai.diff - mot.diff) * 100 < 1e-6,
          f"{abs(hai.diff - mot.diff) * 100:.2e} bp")

    # Chiều: OAS dương làm chân full chiết khấu sâu hơn -> full nhỏ đi.
    duong = price_bond_layered(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par,
                               delta_full=0.005, delta_straight=0.0)
    check(f"{bond_id}: OAS dương làm full giảm, straight giữ nguyên",
          duong.full_price < mot.full_price
          and duong.straight_price.hex() == mot.straight_price.hex(),
          f"full {mot.full_price:.4f} -> {duong.full_price:.4f}")


print("\n3 · Hai cây cùng hình học lưới")
spec = _spec(MAU[0])
zyc, ref, par = _inputs(spec)
ba, ta = build_tree(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par, disc_delta=0.0)
bb, tb = build_tree(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par, disc_delta=0.02)
check("lưới ngày trùng bit", np.array_equal(ba.days, bb.days))
fa, fb = ta.lat.factors[0], tb.lat.factors[0]
check("bước lưới không gian dx trùng bit", np.array_equal(fa.dx, fb.dx))
check("mức trạng thái x trùng bit",
      all(np.array_equal(u, v) for u, v in zip(fa.x, fb.x)))
check("xác suất nhánh trùng bit",
      all(np.array_equal(u, v) for u, v in zip(fa.prob, fb.prob)))
check("chỉ phi / step_discount khác nhau",
      not np.allclose(ta.phi, tb.phi))


print("\n4 · Guard")
try:
    price_bond_layered(spec, VALUE_DATE, disc_df=zyc, ref_df=ref, params=par,
                       delta_full=0.0, delta_straight=0.0, disc_delta=0.01)
    check("truyền nhầm disc_delta -> raise", False)
except TypeError:
    check("truyền nhầm disc_delta -> raise", True)

print("\n" + ("TẤT CẢ ĐỀU PASS" if not _fails else f"{len(_fails)} PHÉP TRƯỢT: {_fails}"))
sys.exit(1 if _fails else 0)
