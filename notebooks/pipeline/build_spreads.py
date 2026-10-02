#%%
"""
Build the **two** spread tables from observed dirty prices, in the required order.

Run BEFORE ``main_v2.py``, whenever either observation file gets new rows::

    cd notebooks/pipeline && python build_spreads.py

Two layers, additive in ``x(t)`` space::

    Stage 1   OAS   = spread of NON-Tier-2 bonds WITH options over the group's LSCK
                      (discount) curve, which is itself built from option-free bonds.
                      Base: the bare group curve.
    Stage 2   delta = additional spread of callable TIER-2 bonds.
                      Base: the group curve WITH OAS ADDED.

Pricing uses four combinations::

    Non-Tier-2   full = OAS          straight = 0
    Tier-2       full = OAS + delta  straight = delta

The bottom row equals "the callable Tier-2 tree **minus** OAS".

**The order is mandatory** (stage 2 uses stage 1's result as its base), so both stages live
in one script rather than two files that must be remembered to run in the right order.

Output goes to ``datasets/spread/``, four files per layer:

``{nontier2_oas,tier2_spread}_monthly.csv``
    The main table. Values are shifts of the **state variable x(t)**, NOT spreads added
    directly to zero rates. ``main_v2.py`` reads exactly these two files.
``*_zero_equivalent.csv``
    Derived table ``· B_a(tau)/tau``: the actual spread on the curve, decaying with tenor.
    Used for reporting and reconciliation, NOT for pricing.
``*_provenance.csv``
    One row per month: the value, ``observed`` or ``carry:<month>``, and the **number of
    observations** and **total par value** behind it. The last two columns must be read
    together: an "observed" month resting on a single trade is not as reliable as a month
    with five trades, and the main table cannot show that difference.
``*_calibrated.csv``
    Content-keyed cache, key ``(bond_id, obs_date, price_hash)``.

Methodology: ``Phuong_phap_luan_spread_trai_phieu_tang_von.html``.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass
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
HOLIDAY_FOLDER_PATH = os.path.join(root, 'datasets', 'holiday')
HULLWHITE_FILE_PATH = os.path.join(root, 'specs', 'hullwhite.json')

# Last month of the spread table. Must equal VALUE_DATE in main_v2.py, otherwise
# `delta_for_bond` reports a missing month.
VALUE_DATE = pd.to_datetime('2026-03-31')

# The SINGLE source of (a, sigma) for every issuer group (TCPH).
#
# Each group's ZYC curve is built as "base curve + margin", and the margins are updated
# irregularly. Calibrating Hull-White directly on a group curve therefore gives unreliable
# (a, sigma), because it also picks up the margin noise. The model takes (a, sigma)
# calibrated on the base curve, while the rate LEVEL is still fitted to the right group
# curve through the Arrow-Debreu forward induction.
#
# In other words: dynamics borrowed from the base curve, level taken from the group curve.
#
# This constant must be the only place that decides this. The behaviour used to be correct
# only because specs/hullwhite.json happened to be overwritten with the same numbers for
# every key: no line of code enforced it, and one recalibration by group name would lose it.
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
# Curves and parameters, loaded once per group
# ---------------------------------------------------------------------------
_curve_cache: dict = {}
_params_cache: dict = {}
_ref_cache: dict = {}


def _hw_params(name):
    """Read ``(a, sigma)`` for ``name`` from specs/hullwhite.json, or raise if missing."""
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
    """
    Return the issuer group's ZYC (LSCK) curve, cached after the first read.

    Args:
        group (str): Issuer group (TCPH) name, e.g. ``LB_G1``.

    Returns:
        pd.DataFrame: The bootstrapped zero curve ``FI_ZYC_VND_<group>.csv``.

    Raises:
        FileNotFoundError: If the curve has not been bootstrapped yet (run main_v2.py once).
    """
    if group not in _curve_cache:
        path = Path(CURVE_FOLDER_PATH) / f"FI_ZYC_VND_{group}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} — chạy main_v2.py một lần để bootstrap đường nhóm."
            )
        _curve_cache[group] = pd.read_csv(path, index_col=0, parse_dates=True)
    return _curve_cache[group]


def ref_curve_df(name):
    """
    Return a reference-rate (LSTC) curve sorted by date, cached after the first read.

    Args:
        name (str): Reference curve name, i.e. the CSV file stem in CURVE_FOLDER_PATH.

    Returns:
        pd.DataFrame: The curve with a DatetimeIndex, sorted ascending.
    """
    if name not in _ref_cache:
        df = pd.read_csv(os.path.join(CURVE_FOLDER_PATH, f'{name}.csv'), index_col=0)
        df.index = pd.to_datetime(df.index)
        _ref_cache[name] = df.sort_index()
    return _ref_cache[name]


def params_of_group(group, ref_curve_name=None):
    """
    Return ``ModelParams`` for a group, including the reference leg for a floating bond.

    ``group`` does NOT determine ``(a_r, sigma_r)``: both come from :data:`BASE_CURVE`. The
    function still takes ``group`` so the signature does not mislead the reader into thinking
    the group is irrelevant: the group still decides the discount *curve*, just not the
    *dynamics*.

    ``rho`` is measured between the base curve and the reference curve, exactly the factor
    pair being simulated. The cache key is ``(group, ref_curve_name)`` to keep the signature
    stable, even though in practice the result depends only on the latter.

    Args:
        group (str): Issuer group (TCPH) name.
        ref_curve_name (str | None): Reference-rate (LSTC) curve name, or None for a fixed-rate
            bond.

    Returns:
        ModelParams: Hull-White parameters for the discount leg, plus ``a_L``, ``sigma_L`` and
        ``rho`` when ``ref_curve_name`` is given.
    """
    key = (group, ref_curve_name)
    if key not in _params_cache:
        a_r, sigma_r = _hw_params(BASE_CURVE)
        if ref_curve_name is None:
            _params_cache[key] = ModelParams(a_r=a_r, sigma_r=sigma_r)
        else:
            a_L, sigma_L = _hw_params(ref_curve_name)
            rho = float(calc_rho(CURVE_NAMES=[BASE_CURVE, ref_curve_name],
                                 value_date=VALUE_DATE).iloc[1, 0])
            _params_cache[key] = ModelParams(a_r=a_r, sigma_r=sigma_r, a_L=a_L,
                                             sigma_L=sigma_L, rho=rho)
    return _params_cache[key]


# ---------------------------------------------------------------------------
def _bao_cao(nhan, prov, zero_eq, a_report, nhom):
    """
    Print the one-row-per-month table, plus the zero-equivalent spread at the report month.

    Args:
        nhan (str): Layer label used in the printed header (e.g. "OAS").
        prov (pd.DataFrame): Provenance table, one row per month.
        zero_eq (pd.DataFrame): Zero-equivalent spread table, months x tenors.
        a_report (float): Mean-reversion speed used for the zero-equivalent conversion.
        nhom (str): Issuer group that ``a_report`` was taken for.
    """
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
    """
    Run one calibration stage: solve per observation -> aggregate by month -> write 4 files.

    Args:
        nhan (str): Layer label for messages ("OAS" or the Tier-2 delta label).
        obs (pd.DataFrame): Price observations for this layer.
        bond_df (pd.DataFrame): Bond term sheet.
        csd (pd.DataFrame): Coupon schedule built from ``bond_df``.
        hol: Holiday calendar.
        tien_to (str): File prefix in SPREAD_FOLDER_PATH (``nontier2_oas`` / ``tier2_spread``).
        since (pd.Timestamp): First month of the table (T1); earlier observations stay in the
            cache only.
        base_delta_of (callable | None): ``(spec, obs_date) -> float`` giving the base shift
            each observation is solved on top of (the OAS in stage 2); None means 0.

    Returns:
        pd.DataFrame: The main monthly table (``<tien_to>_monthly.csv``).
    """
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

    # Observations before T1 stay in the cache for reconciliation but do NOT enter the table.
    # In stage 2 their delta is solved on the bare group curve (no OAS for that month yet),
    # so it is a DIFFERENT quantity: do not compare it with deltas from T1 onwards.
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
    """
    Build the OAS table (stage 1) and then the Tier-2 delta table on top of it (stage 2).

    Without non-Tier-2 observations, stage 1 is skipped and delta is solved on the bare group
    curve, as before the layered architecture.

    Returns:
        int: Process exit code; 1 if there are no Tier-2 observations, else 0.
    """
    bond_df = pd.read_csv(os.path.join(BOND_FOLDER_PATH, 'term_sheet.csv'))
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
        # No base layer yet: run exactly as before, delta solved directly on the group curve
        # and OAS treated as 0. Said loudly, not silently, because delta then means something
        # quite different.
        print("\n[!] nontier2_price_obs.csv chưa có dòng nào.")
        print("    Bỏ qua chặng OAS; delta tăng vốn dò trên đường nhóm trần như")
        print("    trước. Nạp giá trái phiếu không tăng vốn có quyền chọn để bật")
        print("    kiến trúc xếp tầng.")
        since = pd.to_datetime(obs_t2["obs_date"]).min()
        _chang("delta tăng vốn", obs_t2, bond_df, csd, hol,
               tien_to='tier2_spread', since=since)
        return 0

    # T1: the common start, recomputed on every run from the data and VALUE_DATE.
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
        """Return the base for a Tier-2 observation: that month's OAS, or 0 before T1."""
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