"""
Risk spread for capital-raising (Tier 2) bonds.

Methodology: ``Phuong_phap_luan_spread_trai_phieu_tang_von.html``.

Mechanism summary — **this is the easiest part of the whole module to misread**::

    R_T2(t,T)  =  R_G(t,T)  +  delta · B_a(tau)/tau

``delta`` is a shift of the Hull-White tree's **state variable x(t)**, NOT a spread added directly
to the zero rate. Because the pass-through of ``x`` to the term rate is ``B_a(tau)/tau`` — equal to
1 at the short end and tending to ``1/(a·tau)`` at the long end — the premium on the curve
**decays with tenor**. With ``delta = 143 bp`` and ``a = 0.1827``: 1.40% at 3M but only 0.26% at
30Y.

Practical consequence: anyone who reads ``0.0143`` in ``tier2_spread_monthly.csv`` and adds it
straight onto the zero curve will be off by tens of bp at the long end, with **no error raised**.
The derived file ``tier2_spread_zero_equivalent.csv`` exists precisely to prevent that.

Observed prices are **dirty** (``abs(dirty_amount)`` is the settlement amount) and
``tree.price()`` returns a dirty PV, so **there is no accrued-interest adjustment anywhere**.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.bond_pricer import BondTermSheet, ModelParams, build_tree, shift_x
from src.create_buffer_yield import TENOR_MONTHS

__all__ = [
    "TENOR_ORDER", "ISSUER_GROUPS",
    "parse_amount", "parse_tier2_flag", "observed_dirty_price",
    "validate_term_sheet", "PRICE_OBS_COLUMNS", "load_price_obs",
    "common_start",
    "tier2_curve", "remaining_tenor_bucket",
    "CalibResult", "calibrate_delta", "calibrate_all", "load_cache",
    "CACHE_COLUMNS", "build_tier2_spread_table",
    "monthly_delta_raw", "carry_forward", "spread_table",
    "zero_equivalent", "load_spread_table", "delta_for_bond", "spec_fingerprint",
]

TENOR_ORDER = list(TENOR_MONTHS) # 3M .. 30Y, 18 pillars
ISSUER_GROUPS = ("LB_G1", "LB_G2", "LB_G3", "LB_G4", "NBFI", "FB")  # Issuer groups (TCPH), used for validation and categorisation

PRICE_MIN, PRICE_MAX = 20.0, 200.0   # % of par; only meant to catch unit errors

_TRUE = {"y", "yes", "true", "1", "tier2", "t"} # strings read as True for the Tier 2 flag
_FALSE = {"n", "no", "false", "0", "normal", "f", "", "nan", "none"} # strings read as False for the Tier 2 flag


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_amount(x) -> float:
    """
    Parse an amount string such as ``"300,000,000,000"`` into a float (``3e11``).

    Tolerates a minus sign and whitespace. ``pd.read_csv(thousands=",")`` is deliberately not used:
    it applies per column and **silently** leaves the whole column as ``object`` if a single cell is
    malformed, so ``float()`` then blows up far away from the cause.

    Args:
        x: The value to parse (number, string, None or NaN).

    Returns:
        float: The parsed amount, or NaN for None/NaN/empty/"nan"/"none".

    Raises:
        ValueError: If the string cannot be read as a number.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "").replace(" ", "")
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"không đọc được số tiền: {x!r}") from exc


def parse_tier2_flag(x) -> bool:
    """
    Parse a value to determine if it indicates a Tier 2 bond.

    Args:
        x: The input value to parse. It can be of type bool, None, float, or str.

    Returns:
        bool:
            - True if the input indicates a Tier 2 bond (e.g., 'Y', 'yes', 'true', '1', 'tier2',
              't').
            - False if the input indicates a non-Tier 2 bond (e.g., 'N', 'no', 'false', '0',
              'normal', 'f', '', 'nan', 'none').

    Raises:
        ValueError: If the input value cannot be interpreted as a Tier 2 flag.
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x) # Coerce to Python bool if input is a boolean type
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False # Treat None or NaN as False
    s = str(x).strip().lower() # Convert to string, strip whitespace, and lowercase for comparison
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(
        f"cờ is_tier2 không đọc được: {x!r}. Dùng Y/N (hoặc 1/0, TRUE/FALSE)."
    )


def observed_dirty_price(dirty_amount, par_value, face) -> float:
    """
    Compute the dirty price per ``face`` units of par.

    The sign of ``dirty_amount`` is the position direction (positive = own issuance, negative =
    bought from another institution) and **carries no price information** — so the absolute value
    of both amounts is taken.

    A ratio of exactly 1 is a valid result, not a sign of missing data.

    Args:
        dirty_amount: Settlement amount including accrued interest (sign ignored).
        par_value: Par amount of the same observation (sign ignored).
        face: Face value the price is quoted on (e.g. 100).

    Returns:
        float: ``abs(dirty_amount) / abs(par_value) * face``.

    Raises:
        ValueError: If ``par_value`` is missing/zero or ``dirty_amount`` is not finite.
    """
    p, n = parse_amount(dirty_amount), parse_amount(par_value)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"par_value không hợp lệ: {par_value!r}")
    if not np.isfinite(p):
        raise ValueError(f"dirty_amount không hợp lệ: {dirty_amount!r}")
    return abs(p) / abs(n) * float(face)


def validate_term_sheet(bond_df: pd.DataFrame) -> None:
    """
    Validate the **schema** of the term sheet DataFrame for Tier 2 bonds.

    Always callable, even before any real prices exist. Deliberately kept separate from
    :func:`validate_price_data`: missing trade prices only block the spread calibration step and
    must not block pricing of ordinary bonds.

    Args:
        bond_df (pd.DataFrame): The term sheet, with ``group``, ``bond_id`` and ``is_tier2``
            columns.

    Raises:
        ValueError: If a row still has the legacy ``group='Tier2'``, a ``group`` is not in
            ``ISSUER_GROUPS``, the ``is_tier2`` column is missing, or an ``is_tier2`` value
            cannot be parsed.
    """
    stale = bond_df[bond_df["group"].astype(str).str.strip() == "Tier2"] # Strip whitespace from 'group' column before comparison
    if len(stale):
        raise ValueError(
            f"{len(stale)} dòng vẫn ghi group='Tier2'. Cột 'group' nay là phân "
            f"nhóm TCPH (LB_G1/LB_G2/LB_G3); dùng cột 'is_tier2' để đánh dấu "
            f"trái phiếu tăng vốn.\n  " + ", ".join(stale["bond_id"].astype(str)[:6])
        ) # Check for any rows with 'group' column set to 'Tier2' and raise an error if found
    bad = sorted(set(bond_df["group"].astype(str).str.strip()) - set(ISSUER_GROUPS)) # Strip whitespace from 'group' column and check for any values not in ISSUER_GROUPS
    if bad:
        raise ValueError(f"group ngoài {ISSUER_GROUPS}: {bad}") # Raise an error if any values in 'group' column are not in ISSUER_GROUPS
    if "is_tier2" not in bond_df.columns:
        raise ValueError("thiếu cột 'is_tier2' trong term_sheet.csv") # Raise an error if 'is_tier2' column is missing from the DataFrame
    bond_df["is_tier2"].map(parse_tier2_flag) # Check that 'is_tier2' column can be parsed as boolean values using parse_tier2_flag function


PRICE_OBS_COLUMNS = ("bond_id", "obs_date", "dirty_amount", "par_value",
                     "coupon_rate")


def load_price_obs(path, bond_df: pd.DataFrame, *,
                   expect_tier2: bool = True) -> pd.DataFrame:
    """
    Load a long-format price observation file (``*_price_obs.csv``).

    One row = one bond on one date. This is the **only source** for the calibration step; the term
    sheet only keeps ``is_tier2`` for pricing at the reporting date.

    ``expect_tier2`` selects the pool: ``True`` for ``tier2_price_obs.csv`` (accepts only
    capital-raising bonds), ``False`` for ``nontier2_price_obs.csv`` (accepts only ordinary bonds).
    The two pools must be kept apart because they measure two different quantities — letting one
    bond slip into the other pool corrupts both numbers with no error raised.

    The long format is required, not merely tidier: calibration runs over a multi-month history,
    and each observation date falls in a different coupon period with a different volume. A single
    term-sheet column can carry only **one** date.

    Columns:

    ``dirty_amount``
        Settlement amount, including accrued interest. The sign is the position direction and is
        ignored. Being exactly equal to ``par_value`` is valid — a dirty price of 100% of par does
        happen, so there is **no** check here that treats a ratio of 100.0 as a sign of data not
        yet loaded.
    ``par_value``
        Par amount of that observation itself. It is also the weight in the tenor-bucket average,
        so it must be the trade volume, not the outstanding amount of the whole bond.
    ``coupon_rate``
        Current coupon-period rate at ``obs_date``, as a decimal (0.0655). Leave blank to infer it
        from the reference curve as today.

    Args:
        path: Path to the observation CSV.
        bond_df (pd.DataFrame): Term sheet providing ``face``, ``group`` and ``is_tier2``.
        expect_tier2 (bool): Which pool the file belongs to (see above).

    Returns:
        pd.DataFrame: Observations joined with ``face`` and ``group`` from the term sheet, plus a
        ``dirty_price`` column per 100 of par, sorted by ``obs_date`` and ``bond_id``.

    Raises:
        ValueError: On missing columns, unknown ``bond_id``, a bond in the wrong pool, duplicate
            ``(bond_id, obs_date)``, a dirty price outside ``[PRICE_MIN, PRICE_MAX]``, or a
            ``coupon_rate`` given in percent.
    """
    obs = pd.read_csv(path, dtype=str).dropna(how="all")
    missing = [c for c in PRICE_OBS_COLUMNS if c not in obs.columns]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}")
    if obs.empty:
        return obs.assign(obs_date=pd.to_datetime([]), dirty_price=[],
                          par_value=[], face=[], group=[], coupon_rate=[])

    obs["bond_id"] = obs["bond_id"].str.strip()
    obs["obs_date"] = pd.to_datetime(obs["obs_date"], format="%m/%d/%Y")

    sheet = bond_df.set_index(bond_df["bond_id"].astype(str).str.strip())
    unknown = sorted(set(obs["bond_id"]) - set(sheet.index))
    if unknown:
        raise ValueError(f"{path}: bond_id không có trong term sheet: {unknown}")
    sai_pool = sorted({b for b in obs["bond_id"]
                       if parse_tier2_flag(sheet.at[b, "is_tier2"]) != expect_tier2})
    if sai_pool:
        can, thua = (("tăng vốn", "thường") if expect_tier2
                     else ("thường", "tăng vốn"))
        raise ValueError(
            f"{path}: {sai_pool} là trái phiếu {thua}, trong khi pool này chỉ "
            f"nhận trái phiếu {can}. Hai pool đo hai đại lượng khác nhau; lẫn "
            f"vào nhau là hỏng cả hai số mà không có lỗi nào báo."
        )

    dup = obs.duplicated(["bond_id", "obs_date"], keep=False)
    if dup.any():
        d = obs.loc[dup, ["bond_id", "obs_date"]].drop_duplicates()
        raise ValueError(f"{path}: trùng (bond_id, obs_date):\n{d.to_string(index=False)}")

    obs["face"] = sheet.loc[obs["bond_id"], "face"].astype(float).to_numpy()
    obs["group"] = sheet.loc[obs["bond_id"], "group"].astype(str).str.strip().to_numpy()
    obs["par_value"] = obs["par_value"].map(parse_amount)
    obs["dirty_price"] = [
        observed_dirty_price(a, n, f)
        for a, n, f in zip(obs["dirty_amount"], obs["par_value"], obs["face"])
    ]
    obs["coupon_rate"] = pd.to_numeric(obs["coupon_rate"], errors="coerce")

    bad = obs[(obs["dirty_price"] < PRICE_MIN) | (obs["dirty_price"] > PRICE_MAX)]
    if len(bad):
        raise ValueError(
            f"{path}: giá dirty ngoài dải [{PRICE_MIN}, {PRICE_MAX}] trên 100 mệnh "
            f"giá — gần như chắc chắn là sai đơn vị giữa dirty_amount và "
            f"par_value:\n{bad[['bond_id', 'obs_date', 'dirty_price']].to_string(index=False)}"
        )
    hot = obs[obs["coupon_rate"].notna() & (obs["coupon_rate"].abs() > 1.0)]
    if len(hot):
        raise ValueError(
            f"{path}: coupon_rate phải là thập phân (0,0655), không phải phần trăm:"
            f"\n{hot[['bond_id', 'obs_date', 'coupon_rate']].to_string(index=False)}"
        )
    return obs.sort_values(["obs_date", "bond_id"]).reset_index(drop=True)



def common_start(obs_base: pd.DataFrame, obs_layer: pd.DataFrame, value_date):
    """
    Find the start date ``T1`` so both tables begin together, counting back from the value date.

    .. code-block:: text

        d_layer = max{ obs_date of the OVERLAY layer, <= value_date }
        T1      = max{ obs_date of the BASE layer,    <= d_layer    }

    Here the base layer is non-capital-raising bonds with options (producing the OAS) and the
    overlay layer is capital-raising bonds with options (producing delta).

    ``T1`` is **the most recent date on which both layers have fresh observations**. The constraint
    ``T1 <= d_layer`` avoids opening the table in a month where only the base is new while the
    overlay has to be carried from an old month — pairing a fresh layer with a stale one.

    ``value_date`` is a dynamic parameter: changing the reporting period or loading more
    observations moves ``T1`` accordingly. No date may be hard-coded.

    Args:
        obs_base (pd.DataFrame): Base-layer observations (``obs_date`` column).
        obs_layer (pd.DataFrame): Overlay-layer observations (``obs_date`` column).
        value_date: The valuation date.

    Returns:
        pd.Timestamp: The common start date ``T1``.

    Raises:
        ValueError: If the overlay has no observation up to ``value_date``, or the base has none up
            to the overlay's latest date.
    """
    vd = pd.Timestamp(value_date)
    d_base = pd.to_datetime(obs_base["obs_date"])
    d_layer = pd.to_datetime(obs_layer["obs_date"])

    layer_ok = d_layer[d_layer <= vd]
    if layer_ok.empty:
        raise ValueError(
            f"không có quan sát nào của tầng chồng tới ngày định giá {vd.date()}"
        )
    d_l = layer_ok.max()

    base_ok = d_base[d_base <= d_l]
    if base_ok.empty:
        raise ValueError(
            f"không có quan sát tầng nền nào tới {d_l.date()} — tầng nền chưa phủ "
            f"được tầng chồng. Nạp thêm quan sát trái phiếu không tăng vốn có "
            f"quyền chọn, hoặc lùi ngày định giá."
        )
    return base_ok.max()


# ---------------------------------------------------------------------------
# Tier 2 curve and tenor buckets
# ---------------------------------------------------------------------------
def tier2_curve(group_curve, delta: float, a: float):
    """
    Build the capital-raising bond discount curve = the group curve with ``x`` shifted by ``delta``.

    This is only the business name for :func:`src.bond_pricer.shift_x`; there is a single
    implementation, not a duplicate.

    Args:
        group_curve: The issuer-group discount curve (LSCK).
        delta (float): Shift of the state variable ``x(t)``.
        a (float): Hull-White mean reversion.

    Returns:
        The shifted discount curve, as returned by ``shift_x``.
    """
    return shift_x(group_curve, a, delta, refine=True)


def remaining_tenor_bucket(obs_date, maturity_date) -> str:
    """
    Return the remaining tenor, snapped to the nearest of the 18 standard pillars.

    Clamps at the edges instead of raising, and takes the **nearest** pillar rather than rounding
    up — rounding up would push a bond with 8.3 years left all the way into the 10Y bucket.

    Ties go to the **shorter** pillar, consistent with patch step 1 (which favours the shorter
    bucket).

    Args:
        obs_date: The observation date.
        maturity_date: The bond maturity date.

    Returns:
        str: The tenor pillar label (e.g. ``"7Y"``).

    Raises:
        ValueError: If maturity is not after the observation date.
    """
    obs = pd.Timestamp(obs_date)
    mat = pd.Timestamp(maturity_date)
    if mat <= obs:
        raise ValueError(f"đáo hạn {mat.date()} không sau ngày quan sát {obs.date()}")
    best, best_key = None, None
    for tenor, months in TENOR_MONTHS.items():
        dist = abs((obs + pd.DateOffset(months=months) - mat).days)
        key = (dist, months)          # tie -> smaller months wins
        if best_key is None or key < best_key:
            best, best_key = tenor, key
    return best


# ---------------------------------------------------------------------------
# Calibrating delta from a single observation
# ---------------------------------------------------------------------------
@dataclass
class CalibResult:
    """
    Result of calibrating ``delta`` for one (bond, observation date).

    Attributes:
        bond_id (str): Bond identifier.
        obs_date (pd.Timestamp): Observation date.
        group (str): Issuer group (TCPH).
        tenor_bucket (str): Remaining-tenor pillar at ``obs_date``.
        par_value (float): Par amount of the observation (the averaging weight).
        obs_dirty_price (float): Observed dirty price per 100 of par.
        delta (float | None): Calibrated shift of ``x(t)``; None if calibration failed.
        status (str): ``"ok"``, ``"unbracketed"``, ``"before_curve_history"``,
            ``"no_future_coupon"`` or ``"error"``.
        n_eval (int): Number of tree pricings performed.
        resid_price (float | None): Price residual at the solution.
        message (str): Diagnostic message on failure.
    """
    bond_id: str
    obs_date: pd.Timestamp
    group: str
    tenor_bucket: str
    par_value: float
    obs_dirty_price: float
    delta: "float | None" = None
    status: str = "ok"
    n_eval: int = 0
    resid_price: "float | None" = None
    message: str = ""

    def as_row(self) -> dict:
        """
        Convert the result to a flat dict row for the cache table.

        Returns:
            dict: The dataclass fields, with ``obs_date`` as an ISO ``YYYY-MM-DD`` string.
        """
        d = asdict(self)
        d["obs_date"] = pd.Timestamp(self.obs_date).date().isoformat()
        return d


def calibrate_delta(spec: BondTermSheet, obs_date, obs_dirty_price, *,
                    disc_df, params: ModelParams, ref_df=None,
                    base_delta: float = 0.0,
                    lo=-0.02, hi=0.05, xtol=1e-7, max_expand=4,
                    option_anchor=None, **tree_kw) -> CalibResult:
    """
    Solve for ``delta`` so that the tree reproduces the observed dirty price.

    Call/put/cap/floor are **fully** enabled — this is an OAS, not a spread on a straight bond.
    ``decompose()`` is not called inside the loop: it costs 6 pricings per step.

    ``f(delta)`` is strictly decreasing (the premium is positive at every tenor when
    ``delta > 0``), so the bracket is widened mechanically and a sign change is then asserted —
    that assertion doubles as the monotonicity test.

    ``base_delta`` is the **base** that ``delta`` is layered on: the tree uses
    ``disc_delta = base_delta + delta``. Because ``shift_x`` is linear in delta, layering is just
    addition. Used for capital-raising bonds, where the base is the group curve **with the OAS
    of non-capital-raising bonds with options already added**. The returned value is ``delta`` —
    the extra layer — not the total.

    Args:
        spec (BondTermSheet): The bond term sheet.
        obs_date: Observation date.
        obs_dirty_price: Observed dirty price per 100 of par.
        disc_df: Discount curve (LSCK) of the issuer group.
        params (ModelParams): Hull-White parameters.
        ref_df: Reference curve for floating legs, or None.
        base_delta (float): Base shift the calibrated ``delta`` is layered on.
        lo (float): Initial lower bracket.
        hi (float): Initial upper bracket.
        xtol (float): Root-finding tolerance passed to ``brentq``.
        max_expand (int): Maximum number of bracket widenings on each side.
        option_anchor: Anchor date of the American exercise ladder; defaults to the issue date.
        **tree_kw: Extra keyword arguments passed to ``build_tree``.

    Returns:
        CalibResult: The result; ``status != "ok"`` on failure (no exception is raised).
    """
    bucket = remaining_tenor_bucket(obs_date, spec.maturity_date)
    res = CalibResult(
        bond_id=spec.bond_id, obs_date=pd.Timestamp(obs_date), group=spec.group,
        tenor_bucket=bucket, par_value=float("nan"),
        obs_dirty_price=float(obs_dirty_price),
    )
    if option_anchor is None:
        option_anchor = spec.issue_date     # fixed American ladder, see bond_pricer docstring

    tree_kw.setdefault("ref_df_raw", ref_df)
    n = [0]

    def f(delta):
        n[0] += 1
        _, tree = build_tree(
            spec, obs_date, disc_df=disc_df, ref_df=ref_df, params=params,
            disc_delta=base_delta + float(delta), option_anchor=option_anchor,
            require_curve_history=True, **tree_kw,
        )
        return tree.price() - obs_dirty_price

    try:
        f_lo, f_hi = f(lo), f(hi)
        for _ in range(max_expand):
            if f_lo > 0:
                break
            lo -= 0.02
            f_lo = f(lo)
        for _ in range(max_expand):
            if f_hi < 0:
                break
            hi += 0.05
            f_hi = f(hi)
        if not (f_lo > 0 > f_hi):
            res.status = "unbracketed"
            res.n_eval = n[0]
            res.message = (
                f"không bracket được trên [{lo:.3f}, {hi:.3f}]: "
                f"f(lo)={f_lo:+.4f}, f(hi)={f_hi:+.4f}; "
                f"giá quan sát {obs_dirty_price:.4f} trên 100 mệnh giá — "
                f"kiểm lại đơn vị của dirty_amount/par_value"
            )
            return res
        delta = brentq(f, lo, hi, xtol=xtol, maxiter=100)
    except ValueError as exc:
        msg = str(exc)
        res.status = ("before_curve_history" if "sớm hơn dòng đầu tiên" in msg
                      else "no_future_coupon" if "coupon" in msg.lower()
                      else "error")
        res.n_eval, res.message = n[0], msg
        return res

    res.delta, res.n_eval = float(delta), n[0]
    res.resid_price = f(delta)
    # No rejection threshold on |delta|: the price source is primary issuance, and the gap
    # between the coupon fixed at issue and the secondary curve can legitimately be large and
    # change sign. A mechanical threshold would cut exactly the most informative observations.
    # Unusual observations are reviewed via the provenance table (`*_provenance.csv`) and the
    # `*_calibrated.csv` file.
    return res


# ---------------------------------------------------------------------------
# Aggregating into a monthly series
# ---------------------------------------------------------------------------
def monthly_delta_raw(calib_df: pd.DataFrame, *, until=None,
                      since=None) -> pd.Series:
    """
    Average ``delta`` weighted by traded par, **one value per month**.

    .. math:: \\bar\\delta_\\mu = \\frac{\\sum_k N_k \\delta_k}{\\sum_k N_k}

    The pool combines all issuer groups: each issuer's own risk is already in the group curve used
    for discounting, so the remaining ``delta`` measures the premium for subordination — a common
    quantity.

    **Not split by tenor bucket.** The previous version averaged by (month × bucket), but the data
    cannot support that level of detail: on the current book only three buckets (5Y, 7Y, 10Y) out
    of the 18 standard buckets have ever traded, and most months have only one or two observations.
    Splitting by bucket then does not produce a real term structure; it just assigns the noise of a
    single trade to one bucket and spreads it to other buckets via the patch rules — i.e. false
    precision.

    The premium's term structure **still exists**, but it comes from the model rather than from
    bucketing: a single ``delta`` on ``x(t)`` produces a premium ``delta·B_a(tau)/tau`` that is
    strictly decreasing in tenor (see the module header).

    The index is a **continuous** ``PeriodIndex('M')`` — months without observations are still
    present as ``NaN`` so that :func:`carry_forward` can see them. ``until`` extends the index to
    the month of the value date; ``since`` trims the start of the table at a given month (see
    :func:`common_start`) instead of at the first month with an observation.

    Observations before ``since`` are still included in ``raw`` and then dropped by ``reindex`` —
    they have been calibrated and stored in the cache, they just do not enter the table.

    Args:
        calib_df (pd.DataFrame): Calibration results (``status``, ``obs_date``, ``par_value``,
            ``delta``).
        until: Optional value date; extends the index to its month.
        since: Optional start date; the table begins at its month.

    Returns:
        pd.Series: Monthly weighted ``delta`` named ``"delta"``, NaN where a month has no data.

    Raises:
        ValueError: If there is no observation with ``status='ok'``.
    """
    ok = calib_df[calib_df["status"] == "ok"].copy()
    if not len(ok):
        raise ValueError("không có quan sát nào status='ok' để dựng bảng spread")
    ok["month"] = pd.PeriodIndex(pd.to_datetime(ok["obs_date"]), freq="M")

    num = ok.assign(w=ok["par_value"] * ok["delta"]).groupby("month")["w"].sum()
    den = ok.groupby("month")["par_value"].sum()
    raw = num / den

    last = max(raw.index) if until is None else pd.Period(pd.Timestamp(until), freq="M")
    first = (min(raw.index) if since is None
             else pd.Period(pd.Timestamp(since), freq="M"))
    # The tail keeps the old rule — `since` only trims the START of the table, never the tail.
    end = max(last, max(raw.index), first)
    months = pd.period_range(first, end, freq="M")
    return raw.reindex(months).rename("delta")


def carry_forward(raw: pd.Series):
    """
    Fill months without observations with the value of the most recent computed period.

    This is the entire remaining patch rule. The previous version had two extra patch steps
    **within the same month** — borrow from the nearest shorter bucket, then from the nearest
    bucket in either direction — but those only existed because the average was then split by
    tenor bucket. Now each month has a single number, so there is no cell left to patch within a
    month: either the month has observations or it does not.

    No backward fill: empty months before the first observation are dropped rather than borrowing
    a number from the future.

    Args:
        raw (pd.Series): Monthly ``delta`` with a continuous ``PeriodIndex``, NaN for gaps.

    Returns:
        tuple[pd.Series, pd.Series]: ``(filled series, source series)``. The source is
        ``observed`` or ``carry:<month>``.

    Raises:
        ValueError: If no month has an observation.
    """
    months = list(raw.index)
    first = next((m for m in months if pd.notna(raw[m])), None)
    if first is None:
        raise ValueError("không tháng nào có quan sát")

    out, prov, last = {}, {}, None
    for m in (x for x in months if x >= first):
        if pd.notna(raw[m]):
            out[m], prov[m], last = float(raw[m]), "observed", m
        else:
            out[m], prov[m] = out[last], f"carry:{last}"
    idx = pd.PeriodIndex(list(out), freq="M")
    return (pd.Series(list(out.values()), index=idx, name="delta"),
            pd.Series(list(prov.values()), index=idx, name="nguon"))


def spread_table(monthly: pd.Series) -> pd.DataFrame:
    """
    Expand a one-``delta``-per-month series into a (month × 18 tenor buckets) table.

    All columns are **identical by design**: the model has only one ``delta`` per month (see
    :func:`monthly_delta_raw`). The table keeps all 18 columns because the consumers —
    :func:`delta_for_bond` and ``main_v2.py`` — look up by each bond's remaining-tenor bucket.

    Args:
        monthly (pd.Series): Monthly ``delta`` indexed by ``PeriodIndex``.

    Returns:
        pd.DataFrame: Month-indexed table with one column per pillar in ``TENOR_ORDER``.
    """
    return pd.DataFrame({t: monthly for t in TENOR_ORDER}, index=monthly.index)


# ---------------------------------------------------------------------------
# Calibrating the whole pool and building the spread table
# ---------------------------------------------------------------------------
CACHE_COLUMNS = ("bond_id", "obs_date", "group", "tenor_bucket", "par_value",
                 "obs_dirty_price", "delta", "status", "n_eval", "resid_price",
                 "message", "price_hash")


def load_cache(path) -> pd.DataFrame:
    """
    Load the calibration cache; if the file does not exist, return an empty frame with the right
    columns.

    Args:
        path: Path to the cache CSV, or a falsy value for no cache.

    Returns:
        pd.DataFrame: The cached rows with ``CACHE_COLUMNS``.

    Raises:
        ValueError: If the cache file is missing any of ``CACHE_COLUMNS``.
    """
    import os
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    df = pd.read_csv(path, dtype={"price_hash": str})
    missing = [c for c in CACHE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: cache thiếu cột {missing}")
    return df



def _obs_keys(df) -> set:
    """Return ``{(bond_id, 'YYYY-MM-DD')}`` — the key shared by the cache and observation file."""
    if not len(df):
        return set()
    return set(zip(df["bond_id"].astype(str).str.strip(),
                   pd.to_datetime(df["obs_date"]).dt.date.astype(str)))


def _stale_keys(cached, obs) -> set:
    """Return keys present in the cache but no longer in the observation set."""
    return _obs_keys(cached) - _obs_keys(obs)


def calibrate_all(obs: pd.DataFrame, bond_df, coupon_schedule_df,
                  holiday_calendar, cfg, *, curve_of_group, params_of,
                  ref_df_of=None, base_delta_of=None, cache_path=None,
                  prune_stale=True, verbose=True) -> pd.DataFrame:
    """
    Calibrate ``delta`` for **every** observation in ``obs``.

    ``curve_of_group(group) -> zyc_df``, ``params_of(spec) -> ModelParams`` and
    ``ref_df_of(spec) -> ref_df | None`` are passed in rather than looked up inside the function:
    this module knows nothing about the curve folder or ``specs/hullwhite.json``, and thanks to
    that the checks can run on hand-built curves.

    ``base_delta_of(spec, obs_date) -> float`` supplies the **base** for each observation; leaving
    it empty means a base of 0 (the bare group curve).

    ``params_of`` takes the **spec**, not the group. The discount curve depends only on the group,
    but ``a_L``, ``sigma_L`` and ``rho`` depend on each bond's reference curve: passing only the
    group would make every floating bond fall back to ``sigma_L = 0`` and the two-factor tree
    blows up immediately.

    **Cache by content, not by timestamp.** The key is ``(bond_id, obs_date)`` plus
    ``price_hash``; changing any input — the observed price, ``a``, ``sigma``, ``step_days``, or
    any cell of the group curve — changes the hash and that row is recalibrated. A cache keyed on
    file timestamps would miss exactly the most dangerous case: a curve edited in place.

    ``prune_stale`` (on by default) keeps the cache file **always describing the current
    observation set**: rows for ``(bond_id, obs_date)`` no longer in ``obs`` are deleted.
    Previously they were kept, so after replacing the observation file the old dataset's rows
    lived on forever — the cache file described an observation set that no longer existed, and
    anyone reading it to rebuild the table would get numbers different from the real table. The
    spread table itself was not wrong because it is built from ``obs``, but an output file that
    lies is reason enough to fix it.

    Set ``prune_stale=False`` when deliberately running on a **subset** of observations and you
    want to keep the rest of the cache.

    Args:
        obs (pd.DataFrame): Observations from :func:`load_price_obs`.
        bond_df: Term sheet DataFrame.
        coupon_schedule_df: Coupon schedule DataFrame.
        holiday_calendar: Holiday calendar used to build the term sheet.
        cfg: Model configuration.
        curve_of_group: ``group -> zyc_df`` discount curve (LSCK) lookup.
        params_of: ``spec -> ModelParams`` lookup.
        ref_df_of: Optional ``spec -> ref_df | None`` reference curve lookup.
        base_delta_of: Optional ``(spec, obs_date) -> float`` base shift lookup.
        cache_path: Path to the cache CSV, or None to disable caching.
        prune_stale (bool): Drop cache rows for observations no longer in ``obs``.
        verbose (bool): Print per-observation progress.

    Returns:
        pd.DataFrame: One row per observation with ``CACHE_COLUMNS``.
    """
    from src.bond_pricer import BondTermSheet

    cached = load_cache(cache_path)
    # Only reuse rows that calibrated SUCCESSFULLY. Caching a failure is self-blinding: after the
    # real cause is fixed, the key is unchanged and it would keep returning the old error.
    ok_cached = cached[cached["status"] == "ok"] if len(cached) else cached
    hit = {(str(r["bond_id"]), str(r["obs_date"]), str(r["price_hash"])): r
           for _, r in ok_cached.iterrows()}

    ids = bond_df["bond_id"].astype(str).str.strip()
    rows, n_hit = [], 0
    for _, o in obs.iterrows():
        bond_id, obs_date = str(o["bond_id"]), pd.Timestamp(o["obs_date"])
        iloc = int(np.flatnonzero(ids.to_numpy() == bond_id)[0])
        spec = BondTermSheet.from_bond_df(
            bond_df, coupon_schedule_df, holiday_calendar, cfg, iloc=iloc)

        group = spec.group
        zyc_df = curve_of_group(group)
        params = params_of(spec)
        ref_df = ref_df_of(spec) if ref_df_of is not None else None
        base = (0.0 if base_delta_of is None
                else float(base_delta_of(spec, obs_date)))

        key_hash = price_hash(float(o["dirty_price"]), group, params,
                              spec.step_days, zyc_df, ref_df, base_delta=base,
                              spec_key=spec_fingerprint(spec))
        key = (bond_id, obs_date.date().isoformat(), key_hash)
        if key in hit:
            row = dict(hit[key])
            n_hit += 1
        else:
            res = calibrate_delta(spec, obs_date, float(o["dirty_price"]),
                                  disc_df=zyc_df, ref_df=ref_df, params=params,
                                  base_delta=base)
            row = res.as_row()
            row["price_hash"] = key_hash
        # par_value belongs to the OBSERVATION, not the term sheet — calibrate_delta cannot see
        # it, so it is assigned here, even when the row comes from the cache.
        row["par_value"] = float(o["par_value"])
        rows.append(row)
        if verbose:
            d = row.get("delta")
            shown = "—" if d is None or (isinstance(d, float) and np.isnan(d)) \
                else f"{float(d)*1e4:+8.1f} bp"
            print(f"  {bond_id:<28s} {obs_date.date()}  {row['tenor_bucket']:>3s}  "
                  f"{shown}  {row['status']}{'  [cache]' if key in hit else ''}")

    out = pd.DataFrame(rows, columns=list(CACHE_COLUMNS))
    if cache_path:
        fresh = out[out["status"] == "ok"]
        if prune_stale:
            # `obs` is the full observation set and `fresh` already covers everything in it that
            # calibrated, so every old row is either replaced or orphaned.
            stale = _stale_keys(cached, obs)
            merged = fresh
        else:
            stale = set()
            keep = cached[~cached.set_index(["bond_id", "obs_date"]).index.isin(
                fresh.set_index(["bond_id", "obs_date"]).index)] if len(cached) else cached
            keep = keep[keep["status"] == "ok"] if len(keep) else keep
            merged = pd.concat([keep, fresh], ignore_index=True)
        merged.to_csv(cache_path, index=False)
        if verbose and stale:
            print(f"  [cache] xoá {len(stale)} dòng của quan sát không còn trong "
                  f"file: {sorted(stale)[:3]}{' ...' if len(stale) > 3 else ''}")
    if verbose:
        ok = int((out["status"] == "ok").sum())
        print(f"  -> {ok}/{len(out)} quan sát dò được, {n_hit} lấy từ cache")
    return out


def build_tier2_spread_table(calib_df: pd.DataFrame, *, until, a, since=None,
                             out_path=None, zero_path=None, prov_path=None):
    """
    Build the monthly spread table from ``calib_df``.

    ``a`` is used only for the derived ``zero_eq`` table — the main table is still ``delta`` in
    ``x(t)`` space.

    ``provenance`` is a one-row-per-month frame carrying the number of observations and total par
    behind that month's figure — read it alongside the main table to know which months are real
    data and how thin they are.

    Args:
        calib_df (pd.DataFrame): Calibration results from :func:`calibrate_all`.
        until: Value date; the table is extended to its month.
        a (float): Hull-White mean reversion used for the zero-equivalent table.
        since: Optional start date (see :func:`common_start`).
        out_path: Optional CSV path for the spread table.
        zero_path: Optional CSV path for the zero-equivalent table.
        prov_path: Optional CSV path for the provenance table.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ``(spread, zero_eq, provenance)``.
    """
    raw = monthly_delta_raw(calib_df, until=until, since=since)
    monthly, nguon = carry_forward(raw)
    spread = spread_table(monthly)
    zero_eq = zero_equivalent(spread, a)

    ok = calib_df[calib_df["status"] == "ok"].copy()
    ok["month"] = pd.PeriodIndex(pd.to_datetime(ok["obs_date"]), freq="M")
    prov = pd.DataFrame({
        "delta": monthly,
        "nguon": nguon,
        "so_quan_sat": ok.groupby("month").size().reindex(monthly.index).fillna(0).astype(int),
        "tong_menh_gia": ok.groupby("month")["par_value"].sum().reindex(monthly.index),
    })

    for df, path in ((spread, out_path), (zero_eq, zero_path), (prov, prov_path)):
        if path:
            df.to_csv(path)
    return spread, zero_eq, prov


# ---------------------------------------------------------------------------
# Lookup and reporting
# ---------------------------------------------------------------------------
def zero_equivalent(spread_df: pd.DataFrame, a: float) -> pd.DataFrame:
    """
    Convert ``delta`` into a spread on the zero rate, ``delta · B_a(tau)/tau``.

    This is the figure that goes into the methodology document and reports — raw ``delta`` is not
    comparable across groups because ``a`` differs.

    Args:
        spread_df (pd.DataFrame): Month × tenor table of ``delta``.
        a (float): Hull-White mean reversion.

    Returns:
        pd.DataFrame: The zero-rate-equivalent spread table, same shape as ``spread_df``.
    """
    from callput import hw_B
    tau = np.array([TENOR_MONTHS[t] / 12.0 for t in spread_df.columns])
    return spread_df * (hw_B(a, tau) / tau)


def load_spread_table(path) -> pd.DataFrame:
    """
    Load a monthly spread table written by :func:`build_tier2_spread_table`.

    Args:
        path: Path to the spread CSV (month index, one column per tenor pillar).

    Returns:
        pd.DataFrame: The table with a monthly ``PeriodIndex`` and columns in ``TENOR_ORDER``.

    Raises:
        ValueError: If a tenor column is missing or any cell is empty (carry_forward not run).
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    missing = [t for t in TENOR_ORDER if t not in df.columns]
    if missing:
        raise ValueError(f"bảng spread thiếu kỳ hạn: {missing}")
    df = df[TENOR_ORDER]
    if df.isna().any().any():
        raise ValueError("bảng spread còn ô rỗng — chưa chạy carry_forward?")
    return df


def delta_for_bond(spread_df: pd.DataFrame, value_date, maturity_date) -> float:
    """
    Return the single ``delta`` for one bond at the value date.

    Args:
        spread_df (pd.DataFrame): Monthly spread table from :func:`load_spread_table`.
        value_date: The valuation date.
        maturity_date: The bond maturity date.

    Returns:
        float: ``delta`` for the value-date month and the bond's remaining-tenor bucket.

    Raises:
        ValueError: If the table does not contain the value-date month.
    """
    m = pd.Period(pd.Timestamp(value_date), freq="M")
    if m not in spread_df.index:
        raise ValueError(
            f"bảng spread không có tháng {m} (có {spread_df.index.min()}.."
            f"{spread_df.index.max()}) — nối dài index khi dựng bảng"
        )
    return float(spread_df.loc[m, remaining_tenor_bucket(value_date, maturity_date)])


_ROOT = Path(__file__).resolve().parents[1]
# Source files whose content decides a calibrated value. Hashing the code itself means no one has
# to remember to bump a version: any edit invalidates the cache (a full recalibration takes a few
# minutes), and a pricing fix can never be hidden behind numbers cached from the old code.
_PRICING_SOURCES = (
    *sorted((_ROOT / "callput").glob("*.py")),
    *(_ROOT / "src" / f for f in ("bond_schedule.py", "bond_pricer.py", "map_curve.py",
                                   "daycount.py", "tier2_spread.py")),
)


def spec_fingerprint(spec: BondTermSheet) -> str:
    """
    Hash everything about one bond, and the pricing code, that can change its calibrated value.

    It covers the term-sheet row, the coupon schedule and reset dates actually built from it (so
    a change in the holiday calendar or rolling_date is caught through its effect), the adjusted
    maturity, the pricing config, and the source of the pricing code.

    Args:
        spec (BondTermSheet): The bond.

    Returns:
        str: SHA-1 hex digest.
    """
    h = hashlib.sha1()
    h.update(spec.frame.to_csv(index=False).encode())
    h.update(pd.util.hash_pandas_object(spec.coupon_schedule, index=False).values.tobytes())
    h.update(repr([str(d) for d in spec.reset_dates]).encode())
    h.update(repr(spec.maturity_date).encode())
    h.update(repr(spec.cfg).encode())
    for path in _PRICING_SOURCES:
        h.update(path.read_bytes())
    return h.hexdigest()


def price_hash(obs_dirty_price, group, params, step_days, zyc_df,
               ref_df=None, base_delta: float = 0.0, spec_key: str = "") -> str:
    """
    Compute the content-based cache key: a hash of **every** input to one calibration.

    The principle is that the key must cover everything that can change the result, because a
    wrong cache raises no error — it returns an old number that still looks plausible.

    ``a_r`` is **mandatory**: it drives both the tree dynamics and the shape of the bump vector
    ``B_a(tau)/tau``, so re-calibrating Hull-White for the group curve makes every old ``delta``
    meaningless.

    ``base_delta`` must be included too: the two pipelines (non-capital-raising OAS and
    capital-raising delta) run on the **same** set of curves and the **same** parameters, differing
    only in the base. Leaving it out of the key makes the two pipelines share cache cells and
    return each other's numbers.

    ``a_L``, ``sigma_L``, ``rho`` and the content of the reference curve must also be included: for
    floating bonds the reference leg drives the cash flows, not just discounting. A key made of
    the group alone would treat two bonds in the same group with different reference curves as
    one.

    Args:
        obs_dirty_price: Observed dirty price.
        group: Issuer group (TCPH).
        params: Model parameters (``a_r``, ``sigma_r``, ``a_L``, ``sigma_L``, ``rho``).
        step_days: Tree time step in days.
        zyc_df: Group discount curve (LSCK).
        ref_df: Reference curve, or None.
        base_delta (float): Base shift the calibration is layered on.
        spec_key (str): :func:`spec_fingerprint` of the bond (term sheet, built schedule, config
            and pricing code). Without it a term-sheet correction or a pricing fix would keep
            returning the old cached value.

    Returns:
        str: SHA-1 hex digest.
    """
    h = hashlib.sha1()
    h.update(spec_key.encode())
    parts = (f"{obs_dirty_price!r}", group, f"{step_days!r}",
             f"{params.a_r!r}", f"{params.sigma_r!r}", f"{params.a_L!r}",
             f"{params.sigma_L!r}", f"{params.rho!r}", f"{base_delta!r}")
    for part in parts:
        h.update(part.encode())
    for df in (zyc_df, ref_df):
        h.update(b"|")
        if df is not None:
            h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()
