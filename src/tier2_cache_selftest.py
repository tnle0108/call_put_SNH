# -*- coding: utf-8 -*-
"""Phép kiểm hợp đồng ghi cache của ``calibrate_all``.

Chạy:  python src/tier2_cache_selftest.py        (từ thư mục gốc)

Tách khỏi ``tier2_spread_selftest.py`` vì phép kiểm ở đây cần thay
``calibrate_delta`` bằng bản giả — không cần đường cong, không cần dựng cây,
chạy trong chưa tới một giây.

Điều được bảo vệ: **tệp cache luôn mô tả đúng bộ quan sát hiện tại**. Trước đây
dòng của ``(bond_id, obs_date)`` không còn trong file quan sát vẫn được giữ, nên
sau khi thay dữ liệu thì dòng của bộ dữ liệu cũ sống mãi trong cache và ai đọc
tệp đó để dựng lại bảng sẽ ra con số khác với bảng thật.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import src.tier2_spread as T  # noqa: E402
from src.bond_pricer import ModelParams  # noqa: E402

_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        _fails.append(name)


# --- thay calibrate_delta bang ban gia, khoi phai dung cay ------------------
def fake_calibrate(spec, obs_date, obs_px, **kw):
    r = T.CalibResult(bond_id=spec["bond_id"], obs_date=pd.Timestamp(obs_date),
                      group="LB_G1", tenor_bucket="7Y", par_value=float("nan"),
                      obs_dirty_price=float(obs_px))
    r.delta, r.status, r.n_eval = 0.01, "ok", 3
    return r


class FakeSpec(dict):
    __getattr__ = dict.get


def fake_from_bond_df(bond_df, csd, hol, cfg, iloc):
    row = bond_df.iloc[iloc]
    return FakeSpec(bond_id=str(row["bond_id"]).strip(), group="LB_G1",
                    ref_curve_name=None, step_days=21.0,
                    maturity_date=pd.Timestamp("2033-01-01"),
                    issue_date=pd.Timestamp("2023-01-01"))


T.calibrate_delta = fake_calibrate
import src.bond_pricer as BP  # noqa: E402
BP.BondTermSheet.from_bond_df = staticmethod(fake_from_bond_df)

SHEET = pd.DataFrame([dict(bond_id=f"B{i}", face=100, group="LB_G1", is_tier2="Y")
                      for i in (1, 2, 3)])


def run(obs, cache, **kw):
    return T.calibrate_all(obs, SHEET, None, None, None,
                           curve_of_group=lambda g: pd.DataFrame({"3M": [0.06]},
                                                                 index=[pd.Timestamp("2020-01-01")]),
                           params_of=lambda s: ModelParams(a_r=0.18, sigma_r=0.017), cache_path=cache,
                           verbose=False, **kw)


def mk(pairs):
    return pd.DataFrame([dict(bond_id=b, obs_date=pd.Timestamp(d),
                              dirty_price=100.0, par_value=1e11) for b, d in pairs])


print("\nHợp đồng ghi cache của calibrate_all")
cache = os.path.join(tempfile.gettempdir(), "cp_a1_cache.csv")
if os.path.exists(cache):
    os.remove(cache)

run(mk([("B1", "2026-01-15"), ("B2", "2026-02-15")]), cache)
c1 = pd.read_csv(cache)
check("lan 1: cache co dung 2 dong", len(c1) == 2, f"{len(c1)}")

# thay hoan toan bo quan sat
run(mk([("B3", "2026-03-15")]), cache)
c2 = pd.read_csv(cache)
check("lan 2 (thay bo quan sat): cache chi con 1 dong", len(c2) == 1, f"{len(c2)}")
check("cache khong con B1/B2", set(c2.bond_id) == {"B3"}, str(sorted(set(c2.bond_id))))

# prune_stale=False giu lai (chay tap con co chu dich)
if os.path.exists(cache):
    os.remove(cache)
run(mk([("B1", "2026-01-15"), ("B2", "2026-02-15")]), cache)
run(mk([("B3", "2026-03-15")]), cache, prune_stale=False)
c3 = pd.read_csv(cache)
check("prune_stale=False van gop them", len(c3) == 3, f"{len(c3)}")

# moi dong trong cache phai ung voi mot quan sat
obs = mk([("B1", "2026-01-15"), ("B2", "2026-02-15")])
run(obs, cache)
c4 = pd.read_csv(cache)
check("moi dong cache deu co quan sat tuong ung",
      T._obs_keys(c4) == T._obs_keys(obs))

os.remove(cache)

# --------------------------------------------------------------------------
# Hai pipeline (OAS khong tang von / delta tang von) chay tren CUNG bo duong
# cong va CUNG bo tham so, chi khac nen. Bo base_delta khoi khoa la chung dung
# chung o cache va tra so cua nhau — sai ma khong co loi nao bao.
print("\nKhoa cache phai phu ca nen base_delta")
from src.tier2_spread import price_hash  # noqa: E402

_p = ModelParams(a_r=0.18, sigma_r=0.017)
_zyc = pd.DataFrame({"3M": [0.06]}, index=[pd.Timestamp("2020-01-01")])


def _h(bd):
    return price_hash(100.0, "LB_G1", _p, 21.0, _zyc, None, base_delta=bd)


check("doi base_delta thi price_hash doi", _h(0.0) != _h(0.01))
check("cung base_delta thi price_hash khong doi", _h(0.01) == _h(0.01))
check("mac dinh base_delta = 0",
      _h(0.0) == price_hash(100.0, "LB_G1", _p, 21.0, _zyc, None))

print("\n" + ("TAT CA DEU PASS" if not _fails else f"{len(_fails)} PHEP TRUOT: {_fails}"))
sys.exit(1 if _fails else 0)
