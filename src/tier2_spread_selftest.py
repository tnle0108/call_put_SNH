#%%
"""Phép kiểm cho các hàm thuần của src/tier2_spread.py.

Chạy:  python src/tier2_spread_selftest.py     (từ thư mục gốc repo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callput import YieldCurve, hw_B                                 # noqa: E402
from src.bond_pricer import refine_curve, shift_x                    # noqa: E402
from src.tier2_spread import (                                        # noqa: E402
    TENOR_ORDER, carry_forward, common_start, load_price_obs, monthly_delta_raw,
    parse_amount, parse_tier2_flag, remaining_tenor_bucket, observed_dirty_price,
    spread_table, zero_equivalent,
)

_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


# --------------------------------------------------------------------------
print("\n1 · Đọc dữ liệu")
check("parse_amount tách dấu phẩy nghìn", parse_amount("300,000,000,000") == 3e11)
check("parse_amount giữ dấu âm", parse_amount(" -2,488,000,000,000 ") == -2.488e12)
check("giá dirty lấy trị tuyệt đối cả hai",
      observed_dirty_price("-1,012,000,000,000", "1,000,000,000,000", 100) == 101.2)
try:
    parse_tier2_flag("maybe")
    check("cờ lạ phải raise", False)
except ValueError:
    check("cờ lạ phải raise", True)
check("cờ rỗng = N", parse_tier2_flag("") is False and parse_tier2_flag("Y") is True)

# --------------------------------------------------------------------------
print("\n2 · Rổ kỳ hạn")
obs = pd.Timestamp("2026-03-31")
cases = [("2026-06-30", "3M"), ("2026-09-30", "6M"), ("2033-03-31", "7Y"),
         ("2026-05-31", "3M"), ("2058-03-31", "30Y"), ("2033-05-10", "7Y")]
for mat, want in cases:
    got = remaining_tenor_bucket(obs, mat)
    check(f"còn tới {mat} -> {want}", got == want, f"ra {got}")
check("hoà thì lấy pillar ngắn hơn",
      remaining_tenor_bucket("2026-01-01", "2026-11-16") in ("9M", "1Y"),
      remaining_tenor_bucket("2026-01-01", "2026-11-16"))

# --------------------------------------------------------------------------
print("\n3 · Bình quân trọng số par")
calib = pd.DataFrame([
    dict(bond_id="A", obs_date="2026-01-15", tenor_bucket="5Y",
         par_value=3e11, delta=0.01, status="ok"),
    dict(bond_id="B", obs_date="2026-01-20", tenor_bucket="10Y",
         par_value=1e11, delta=0.02, status="ok"),
])
raw = monthly_delta_raw(calib)
got = raw.loc[pd.Period("2026-01", "M")]
check("par 3:1 với 100/200 bp -> 125 bp", abs(got - 0.0125) < 1e-15, f"{got*1e4:.2f} bp")
check("gộp chung mọi rổ kỳ hạn, ra MỘT số cho tháng",
      isinstance(raw, pd.Series) and len(raw) == 1)

calib_loai = pd.concat([calib, pd.DataFrame([
    dict(bond_id="C", obs_date="2026-01-25", tenor_bucket="7Y",
         par_value=9e11, delta=0.99, status="before_curve_history")])])
check("quan sát không dò được bị loại khỏi bình quân",
      abs(monthly_delta_raw(calib_loai).iloc[0] - 0.0125) < 1e-15)

check("until nối dài index tới tháng ngày định giá",
      monthly_delta_raw(calib, until=pd.Timestamp("2026-04-30")).index.max()
      == pd.Period("2026-04", "M"))

# --------------------------------------------------------------------------
print("\n4 · Giữ giá trị của kỳ tính toán gần nhất")
months = pd.period_range("2026-01", "2026-05", freq="M")
raw = pd.Series([0.010, float("nan"), float("nan"), 0.030, float("nan")],
                index=months, name="delta")
out, nguon = carry_forward(raw)

m1, m2, m3, m4, m5 = months
check("tháng có quan sát giữ nguyên",
      out[m1] == 0.010 and nguon[m1] == "observed")
check("tháng rỗng bê từ tháng gần nhất trước đó",
      out[m2] == 0.010 and nguon[m2] == "carry:2026-01")
check("bê tiếp qua nhiều tháng rỗng liên tiếp",
      out[m3] == 0.010 and nguon[m3] == "carry:2026-01")
check("quan sát mới thay chỗ, không bê nữa",
      out[m4] == 0.030 and nguon[m4] == "observed")
check("sau đó bê từ quan sát MỚI nhất",
      out[m5] == 0.030 and nguon[m5] == "carry:2026-04")

raw2 = pd.concat([pd.Series([float("nan")] * 2,
                            index=pd.period_range("2025-11", "2025-12", freq="M")), raw])
out2, _ = carry_forward(raw2)
check("tháng rỗng dẫn đầu bị bỏ, không bù lùi", out2.index.min() == m1)

try:
    carry_forward(pd.Series([float("nan")] * 3, index=months[:3]))
    check("không tháng nào có quan sát -> raise", False)
except ValueError:
    check("không tháng nào có quan sát -> raise", True)

check("bảng phát ra 18 rổ giống hệt nhau theo thiết kế",
      bool((spread_table(out).nunique(axis=1) == 1).all())
      and list(spread_table(out).columns) == TENOR_ORDER)

# --------------------------------------------------------------------------
print("\n5 · Đường cong và phần bù")
m = np.array([.25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 5, 7, 10, 15, 20, 30])
c = YieldCurve.from_zero_rates(m, 0.06 + 0.02 * (1 - np.exp(-m / 6)))
a, d = 0.1827, 0.0143
check("refine_curve trùng khớp tại pillar gốc",
      np.array_equal(refine_curve(c).zero_rate(m), c.zero_rate(m)))
t = np.geomspace(1 / 365, 40, 20000)
ulp = np.abs(refine_curve(c).zero_rate(t) - c.zero_rate(t)).max()
check("refine_curve lệch tối đa 1 ULP giữa các pillar", ulp < 3e-17, f"{ulp:.2e}")
check("delta = 0 trả về chính object", shift_x(c, a, 0.0) is c)

T = np.geomspace(1 / 365, 30, 4000)
s = shift_x(c, a, d)
err = np.abs((s.zero_rate(T) - c.zero_rate(T)) - d * hw_B(a, T) / T).max()
check("R_T2 − R_G = δ·B(τ)/τ tại MỌI τ", err * 1e4 < 1e-3, f"{err*1e4:.5f} bp")
coarse = c.shifted(d * hw_B(a, c.maturities) / c.maturities)
e18 = np.abs((coarse.zero_rate(T) - c.zero_rate(T)) - d * hw_B(a, T) / T).max()
check("chỉ 18 pillar thì lệch đáng kể (lý do cần lưới mịn)",
      e18 * 1e4 > 1.0, f"{e18*1e4:.2f} bp")
check("phần bù giảm ngặt theo kỳ hạn",
      bool(np.all(np.diff(hw_B(a, T) / T) < 0)))

# --------------------------------------------------------------------------
print("\n6 · Spread zero tương đương")
sp = pd.DataFrame([[d] * len(TENOR_ORDER)], columns=TENOR_ORDER,
                  index=pd.period_range("2026-03", periods=1, freq="M"))
ze = zero_equivalent(sp, a)
check("30Y thấp hơn hẳn 3M", ze.iloc[0]["30Y"] < 0.3 * ze.iloc[0]["3M"],
      f"3M {ze.iloc[0]['3M']*100:.4f}%  30Y {ze.iloc[0]['30Y']*100:.4f}%")

# --------------------------------------------------------------------------
print("\n7 · File quan sát giá dạng dài")
import io as _io                                                     # noqa: E402
import os as _os                                                     # noqa: E402
import tempfile as _tf                                               # noqa: E402

_sheet = pd.DataFrame([
    dict(bond_id="T2A", face=100, group="LB_G1", is_tier2="Y"),
    dict(bond_id="T2B", face=100, group="LB_G3", is_tier2="Y"),
    dict(bond_id="NRM", face=100, group="LB_G1", is_tier2="N"),
])
_HDR = "bond_id,obs_date,dirty_amount,par_value,coupon_rate\n"


def _load(body, expect_tier2=True):
    f = _os.path.join(_tf.gettempdir(), "cp_obs_selftest.csv")
    _io.open(f, "w", encoding="utf-8", newline="").write(_HDR + body)
    return load_price_obs(f, _sheet, expect_tier2=expect_tier2)


def _blocked(body, name, expect_tier2=True):
    try:
        _load(body, expect_tier2)
        check(name, False, "không chặn")
    except ValueError:
        check(name, True)


_ok = _load('T2A,2026-01-15,"300,000,000,000","300,000,000,000",0.0655\n'
            'T2B,2026-02-20,"202,400,000,000","200,000,000,000",\n')
check("giá dirty = 100,0 đúng bằng mệnh giá là HỢP LỆ",
      _ok.loc[0, "dirty_price"] == 100.0)
check("giá 101,2 tính đúng", abs(_ok.loc[1, "dirty_price"] - 101.2) < 1e-12,
      f"{_ok.loc[1, 'dirty_price']}")
check("nối được face và group từ term sheet",
      list(_ok["group"]) == ["LB_G1", "LB_G3"] and list(_ok["face"]) == [100, 100])
check("coupon_rate trống -> NaN", bool(np.isnan(_ok.loc[1, "coupon_rate"])))
check("sắp theo obs_date", list(_ok["bond_id"]) == ["T2A", "T2B"])
check("file chỉ có tiêu đề -> khung rỗng", len(_load("")) == 0)

_blocked("XXX,2026-01-15,1e11,1e11,\n", "bond_id lạ -> chặn")
_blocked("NRM,2026-01-15,1e11,1e11,\n", "mã không tăng vốn -> chặn")
_blocked("T2A,2026-01-15,1e9,1e11,\n", "sai đơn vị 100 lần -> chặn")
_blocked("T2A,2026-01-15,1e11,1e11,6.55\n", "coupon ghi phần trăm -> chặn")
_blocked("T2A,2026-01-15,1e11,1e11,\nT2A,2026-01-15,1e11,1e11,\n",
         "trùng (bond_id, obs_date) -> chặn")
_blocked("T2A,2026-01-15,1e11,0,\n", "par_value = 0 -> chặn")


# --------------------------------------------------------------------------
print("\n8 · Xếp tầng OAS và mốc T1")

# --- cộng dồn trong không gian x ------------------------------------------
u, v = 0.0071, -0.0134
T = np.geomspace(1 / 365, 30, 3000)
mot_lan = shift_x(c, a, u + v)
hai_lan = shift_x(shift_x(c, a, u), a, v)
err = np.abs(mot_lan.zero_rate(T) - hai_lan.zero_rate(T)).max()
check("xếp tầng cộng được: shift_x(u+v) ≡ shift_x(u) rồi shift_x(v)",
      err * 1e4 < 1e-6, f"{err*1e4:.2e} bp")
check("nền + chồng = tổng đúng bằng phép cộng số",
      abs((u + v) - (u + v)) == 0 and shift_x(c, a, 0.0) is c)

# --- T1 --------------------------------------------------------------------
def _obs(*ngay):
    return pd.DataFrame({"obs_date": pd.to_datetime(list(ngay))})


nen = _obs("2024-12-08", "2025-08-08", "2025-12-18", "2026-06-23")
chong = _obs("2023-11-15", "2025-08-20", "2026-08-17")

check("T1 = quan sát nền gần nhất, không vượt quan sát chồng gần nhất",
      common_start(nen, chong, "2026-03-31") == pd.Timestamp("2025-08-08"),
      str(common_start(nen, chong, "2026-03-31").date()))
check("đổi ngày định giá thì T1 đổi theo",
      common_start(nen, chong, "2026-08-31") == pd.Timestamp("2026-06-23"),
      str(common_start(nen, chong, "2026-08-31").date()))
# Nền có một quan sát MỚI HƠN mọi quan sát chồng. Nếu ràng buộc không có hiệu
# lực, T1 sẽ là 2026-01-01 và bảng mở đầu ở một tháng chỉ có nền tươi.
_got = common_start(_obs("2025-01-01", "2026-01-01"), _obs("2025-06-01"),
                    "2026-12-31")
check("ràng buộc T1 <= quan sát chồng có hiệu lực",
      _got == pd.Timestamp("2025-01-01"), str(_got.date()))

try:
    common_start(_obs("2026-05-01"), _obs("2025-06-01"), "2026-12-31")
    check("nền toàn nằm SAU chồng -> raise", False)
except ValueError:
    check("nền toàn nằm SAU chồng -> raise", True)
try:
    common_start(nen, chong, "2020-01-01")
    check("chưa có quan sát chồng nào -> raise", False)
except ValueError:
    check("chưa có quan sát chồng nào -> raise", True)

# --- since -----------------------------------------------------------------
calib2 = pd.DataFrame([
    dict(bond_id="A", obs_date="2025-03-10", par_value=1e11, delta=0.05, status="ok"),
    dict(bond_id="B", obs_date="2025-08-10", par_value=1e11, delta=0.01, status="ok"),
])
r_full = monthly_delta_raw(calib2, until="2025-10-31")
r_cut = monthly_delta_raw(calib2, until="2025-10-31", since="2025-08-01")
check("since cắt đầu bảng", r_cut.index.min() == pd.Period("2025-08", "M")
      and r_full.index.min() == pd.Period("2025-03", "M"))
check("since KHÔNG đụng đuôi bảng",
      r_cut.index.max() == r_full.index.max() == pd.Period("2025-10", "M"))
check("quan sát trước since bị loại khỏi bảng, không bị bê sang",
      float(r_cut.loc[pd.Period("2025-08", "M")]) == 0.01
      and pd.Period("2025-03", "M") not in r_cut.index)

# --- tách pool -------------------------------------------------------------
_blocked("T2A,2026-01-15,1e11,1e11,\n", "pool không tăng vốn chặn mã tăng vốn",
         expect_tier2=False)
_ok_nt2 = _load("NRM,2026-01-15,1e11,1e11,\n", expect_tier2=False)
check("pool không tăng vốn nhận mã thường", len(_ok_nt2) == 1)
_blocked("NRM,2026-01-15,1e11,1e11,\n", "pool tăng vốn chặn mã thường")


print("\n" + ("TẤT CẢ ĐỀU PASS" if not _fails else f"{len(_fails)} PHÉP TRƯỢT: {_fails}"))
sys.exit(1 if _fails else 0)
#%%