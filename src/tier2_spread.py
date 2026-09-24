"""Spread rủi ro cho trái phiếu tăng vốn (Tier 2).

Phương pháp luận: ``Phuong_phap_luan_spread_trai_phieu_tang_von.html``.

Tóm tắt cơ chế — **đây là chỗ dễ hiểu nhầm nhất của cả module**::

    R_T2(t,T)  =  R_G(t,T)  +  delta · B_a(tau)/tau

``delta`` là lượng dịch **biến trạng thái x(t)** của cây Hull-White, KHÔNG phải
một spread cộng thẳng vào lãi suất zero.  Vì hệ số truyền của ``x`` sang lãi suất
kỳ hạn là ``B_a(tau)/tau`` — bằng 1 ở đầu ngắn, tiến về ``1/(a·tau)`` ở đầu dài —
phần bù trên đường cong **suy giảm theo kỳ hạn**.  Với ``delta = 143 bp`` và
``a = 0,1827``: 1,40% ở 3M nhưng chỉ 0,26% ở 30Y.

Hệ quả thực tế: ai đọc ``0.0143`` trong ``tier2_spread_monthly.csv`` rồi cộng
thẳng vào đường zero sẽ sai vài chục bp ở đầu dài mà **không có lỗi nào báo**.
File phái sinh ``tier2_spread_zero_equivalent.csv`` tồn tại chính để tránh việc đó.

Giá quan sát là giá **dirty** (``abs(dirty_amount)`` là số tiền thanh toán) và
``tree.price()`` trả PV dirty, nên **không có điều chỉnh lãi dự thu ở bất kỳ đâu**.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

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
    "zero_equivalent", "load_spread_table", "delta_for_bond",
]

TENOR_ORDER = list(TENOR_MONTHS)            # 3M .. 30Y, 18 pillar
ISSUER_GROUPS = ("LB_G1", "LB_G2", "LB_G3")

PRICE_MIN, PRICE_MAX = 20.0, 200.0   # % mệnh giá; chỉ để bắt sai đơn vị

_TRUE = {"y", "yes", "true", "1", "tier2", "t"}
_FALSE = {"n", "no", "false", "0", "normal", "f", "", "nan", "none"}


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def parse_amount(x) -> float:
    """``"300,000,000,000"`` -> ``3e11``.  Chịu được dấu âm và khoảng trắng.

    Không dùng ``pd.read_csv(thousands=",")``: nó áp theo cột và **im lặng** để
    nguyên cột dưới dạng ``object`` nếu một ô hỏng, khiến ``float()`` nổ ở chỗ
    cách xa nguyên nhân.
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
    """Cờ tăng vốn, **nghiêm ngặt**.

    Token lạ thì raise chứ không mặc định ``False``: mặc định âm thầm chính là
    lỗi mà cả thay đổi này sinh ra để loại bỏ — một trái phiếu tăng vốn bị định
    giá trên đường cong thường mà không ai biết.
    """
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(
        f"cờ is_tier2 không đọc được: {x!r}. Dùng Y/N (hoặc 1/0, TRUE/FALSE)."
    )


def observed_dirty_price(dirty_amount, par_value, face) -> float:
    """Giá dirty trên ``face`` đơn vị mệnh giá.

    Dấu của ``dirty_amount`` là chiều vị thế (dương = tự phát hành, âm = mua của
    tổ chức khác), **không mang thông tin giá** — nên lấy trị tuyệt đối cả hai.

    Tỷ lệ bằng đúng 1 là kết quả hợp lệ, không phải dấu hiệu thiếu dữ liệu.
    """
    p, n = parse_amount(dirty_amount), parse_amount(par_value)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"par_value không hợp lệ: {par_value!r}")
    if not np.isfinite(p):
        raise ValueError(f"dirty_amount không hợp lệ: {dirty_amount!r}")
    return abs(p) / abs(n) * float(face)


def validate_term_sheet(bond_df: pd.DataFrame) -> None:
    """Kiểm **lược đồ** term sheet. Luôn gọi được, kể cả khi chưa có giá thật.

    Tách khỏi :func:`validate_price_data` có chủ đích: thiếu giá giao dịch chỉ
    chặn bước hiệu chỉnh spread, không được chặn việc định giá trái phiếu thường.
    """
    stale = bond_df[bond_df["group"].astype(str).str.strip() == "Tier2"]
    if len(stale):
        raise ValueError(
            f"{len(stale)} dòng vẫn ghi group='Tier2'. Cột 'group' nay là phân "
            f"nhóm TCPH (LB_G1/LB_G2/LB_G3); dùng cột 'is_tier2' để đánh dấu "
            f"trái phiếu tăng vốn.\n  " + ", ".join(stale["bond_id"].astype(str)[:6])
        )
    bad = sorted(set(bond_df["group"].astype(str).str.strip()) - set(ISSUER_GROUPS))
    if bad:
        raise ValueError(f"group ngoài {ISSUER_GROUPS}: {bad}")
    if "is_tier2" not in bond_df.columns:
        raise ValueError("thiếu cột 'is_tier2' trong bond placeholder.csv")
    bond_df["is_tier2"].map(parse_tier2_flag)      # nổ sớm nếu có token lạ


PRICE_OBS_COLUMNS = ("bond_id", "obs_date", "dirty_amount", "par_value",
                     "coupon_rate")


def load_price_obs(path, bond_df: pd.DataFrame, *,
                   expect_tier2: bool = True) -> pd.DataFrame:
    """Đọc file quan sát giá dạng dài (``*_price_obs.csv``).

    Một dòng = một trái phiếu tại một ngày.  Đây là **nguồn duy nhất** cho bước
    hiệu chỉnh; term sheet chỉ còn giữ ``is_tier2`` để định giá tại kỳ báo cáo.

    ``expect_tier2`` chọn pool: ``True`` cho ``tier2_price_obs.csv`` (chỉ nhận mã
    tăng vốn), ``False`` cho ``nontier2_price_obs.csv`` (chỉ nhận mã thường).  Hai
    pool phải tách bạch vì chúng đo hai đại lượng khác nhau — lẫn một mã sang pool
    kia là hỏng cả hai số mà không có lỗi nào báo.

    Dạng dài là bắt buộc chứ không phải cho gọn: hiệu chỉnh chạy trên lịch sử
    nhiều tháng, mà mỗi ngày quan sát rơi vào một kỳ coupon khác nhau và một
    khối lượng khác nhau.  Một cột trong term sheet chỉ chở được **một** ngày.

    Cột:

    ``dirty_amount``
        Số tiền thanh toán, đã gồm lãi dự thu.  Dấu là chiều vị thế và bị bỏ
        qua.  Bằng đúng ``par_value`` là hợp lệ — giá dirty 100% mệnh giá xảy ra
        thật, nên ở đây **không** có phép kiểm nào coi tỷ lệ 100,0 là dấu hiệu
        dữ liệu chưa nạp.
    ``par_value``
        Mệnh giá của chính quan sát đó.  Cũng là trọng số bình quân theo rổ kỳ
        hạn, nên phải là khối lượng của giao dịch chứ không phải dư nợ cả mã.
    ``coupon_rate``
        Lãi suất kỳ coupon hiện hành tại ``obs_date``, dạng thập phân (0,0655).
        Để trống thì suy từ đường tham chiếu như hiện nay.

    Trả về khung đã nối ``face`` và ``group`` từ term sheet, thêm cột
    ``dirty_price`` quy về 100 mệnh giá.
    """
    obs = pd.read_csv(path, dtype=str).dropna(how="all")
    missing = [c for c in PRICE_OBS_COLUMNS if c not in obs.columns]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}")
    if obs.empty:
        return obs.assign(obs_date=pd.to_datetime([]), dirty_price=[],
                          par_value=[], face=[], group=[], coupon_rate=[])

    obs["bond_id"] = obs["bond_id"].str.strip()
    obs["obs_date"] = pd.to_datetime(obs["obs_date"], format="mixed")

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
    """Mốc ``T1`` để hai bảng cùng bắt đầu, tính lùi từ ngày định giá.

    .. code-block:: text

        d_layer = max{ obs_date của tầng CHỒNG, <= value_date }
        T1      = max{ obs_date của tầng NỀN,   <= d_layer    }

    Ở đây tầng nền là trái phiếu không tăng vốn có quyền chọn (sinh ra OAS), tầng
    chồng là trái phiếu tăng vốn có quyền chọn (sinh ra delta).

    ``T1`` là **lần gần nhất mà cả hai tầng đều có quan sát tươi**.  Ràng buộc
    ``T1 <= d_layer`` tránh mở bảng ở một tháng chỉ có nền mới trong khi tầng
    chồng phải bê từ tháng cũ — ghép một tầng tươi với một tầng ôi.

    ``value_date`` là tham số động: đổi kỳ báo cáo hoặc nạp thêm quan sát thì
    ``T1`` đổi theo.  Không được ghi cứng ngày nào vào code.
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
# Đường cong Tier 2 và rổ kỳ hạn
# ---------------------------------------------------------------------------
def tier2_curve(group_curve, delta: float, a: float, *, n_dense: int = 1500):
    """Đường chiết khấu của trái phiếu tăng vốn = đường nhóm với ``x`` dịch ``delta``.

    Chỉ là tên gọi theo nghiệp vụ của :func:`src.bond_pricer.shift_x`; một hiện
    thực duy nhất, không nhân bản.
    """
    return shift_x(group_curve, a, delta, refine=True)


def remaining_tenor_bucket(obs_date, maturity_date) -> str:
    """Kỳ hạn còn lại, snap về pillar gần nhất trong 18 pillar chuẩn.

    Kẹp biên thay vì raise, và lấy pillar **gần nhất** chứ không làm tròn lên —
    làm tròn lên đẩy một trái phiếu còn 8,3 năm vào hẳn rổ 10Y.

    Hoà thì lấy pillar **ngắn hơn**, nhất quán với bước vá 1 (vốn ưu tiên rổ ngắn).
    """
    obs = pd.Timestamp(obs_date)
    mat = pd.Timestamp(maturity_date)
    if mat <= obs:
        raise ValueError(f"đáo hạn {mat.date()} không sau ngày quan sát {obs.date()}")
    best, best_key = None, None
    for tenor, months in TENOR_MONTHS.items():
        dist = abs((obs + pd.DateOffset(months=months) - mat).days)
        key = (dist, months)          # hoà -> months nhỏ hơn thắng
        if best_key is None or key < best_key:
            best, best_key = tenor, key
    return best


# ---------------------------------------------------------------------------
# Hiệu chỉnh delta từ một quan sát
# ---------------------------------------------------------------------------
@dataclass
class CalibResult:
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
        d = asdict(self)
        d["obs_date"] = pd.Timestamp(self.obs_date).date().isoformat()
        return d


def calibrate_delta(spec: BondTermSheet, obs_date, obs_dirty_price, *,
                    disc_df, params: ModelParams, ref_df=None,
                    base_delta: float = 0.0,
                    lo=-0.02, hi=0.05, xtol=1e-7, max_expand=4,
                    option_anchor=None, **tree_kw) -> CalibResult:
    """Dò ``delta`` sao cho cây tái tạo đúng giá dirty quan sát được.

    Bật **đầy đủ** call/put/cap/floor — đây là OAS, không phải spread trên trái
    phiếu straight.  Không gọi ``decompose()`` trong vòng lặp: nó tốn 6 lần định
    giá mỗi bước.

    ``f(delta)`` giảm ngặt (phần bù dương ở mọi kỳ hạn khi ``delta > 0``), nên
    bracket nới cơ học rồi khẳng định đổi dấu — bước khẳng định đó đồng thời là
    phép thử đơn điệu.

    ``base_delta`` là **nền** mà ``delta`` chồng lên: cây dùng
    ``disc_delta = base_delta + delta``.  Vì ``shift_x`` tuyến tính theo delta,
    xếp tầng chỉ là cộng số.  Dùng cho trái phiếu tăng vốn, nơi nền là đường
    nhóm **đã cộng OAS** của trái phiếu không tăng vốn có quyền chọn.  Giá trị
    trả về là ``delta`` — phần chồng thêm — chứ không phải tổng.
    """
    bucket = remaining_tenor_bucket(obs_date, spec.maturity_date)
    res = CalibResult(
        bond_id=spec.bond_id, obs_date=pd.Timestamp(obs_date), group=spec.group,
        tenor_bucket=bucket, par_value=float("nan"),
        obs_dirty_price=float(obs_dirty_price),
    )
    if option_anchor is None:
        option_anchor = spec.issue_date     # thang American cố định, xem docstring bond_pricer

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
    # Không đặt ngưỡng loại |delta|: nguồn giá là phát hành sơ cấp, và chênh
    # giữa coupon ấn định lúc phát hành với đường cong thứ cấp có thể lớn và
    # đổi dấu một cách chính đáng. Một ngưỡng cơ học sẽ cắt đúng những quan sát
    # mang nhiều thông tin nhất. Quan sát bất thường được nhìn qua bảng nguồn
    # (`*_provenance.csv`) và tệp `*_calibrated.csv`.
    return res


# ---------------------------------------------------------------------------
# Gộp thành chuỗi tháng
# ---------------------------------------------------------------------------
def monthly_delta_raw(calib_df: pd.DataFrame, *, until=None,
                      since=None) -> pd.Series:
    """Bình quân ``delta`` theo mệnh giá giao dịch, **một giá trị cho mỗi tháng**.

    .. math:: \\bar\\delta_\\mu = \\frac{\\sum_k N_k \\delta_k}{\\sum_k N_k}

    Pool gộp chung mọi phân nhóm tổ chức phát hành: phần rủi ro riêng của từng
    tổ chức đã nằm trong đường cong nhóm dùng để chiết khấu, nên ``delta`` còn
    lại đo phần bù của tính chất thứ cấp — đại lượng chung.

    **Không tách theo rổ kỳ hạn.**  Bản trước chia bình quân theo (tháng × rổ),
    nhưng dữ liệu không đỡ nổi mức chi tiết đó: trên sổ hiện tại chỉ ba rổ
    (5Y, 7Y, 10Y) từng có giao dịch trong số 18 rổ chuẩn, và phần lớn tháng chỉ
    có một tới hai quan sát. Chia theo rổ khi đó không tạo ra cấu trúc kỳ hạn
    thật mà chỉ gán nhiễu của một giao dịch lẻ vào một rổ rồi lan sang các rổ
    khác qua quy tắc vá — tức là độ chính xác giả.

    Cấu trúc kỳ hạn của phần bù **vẫn còn**, nhưng đến từ mô hình chứ không từ
    việc chia rổ: một ``delta`` duy nhất trên ``x(t)`` sinh ra phần bù
    ``delta·B_a(tau)/tau`` giảm ngặt theo kỳ hạn (xem đầu module).

    Index là ``PeriodIndex('M')`` **liên tục** — tháng không có quan sát vẫn hiện
    diện dưới dạng ``NaN`` để :func:`carry_forward` nhìn thấy.  ``until`` nối dài
    index tới tháng của ngày định giá; ``since`` cắt đầu bảng tại một tháng cho
    trước (xem :func:`common_start`) thay vì tại tháng đầu tiên có quan sát.

    Quan sát nằm trước ``since`` vẫn được tính vào ``raw`` rồi bị ``reindex`` loại
    ra — chúng đã được dò và lưu trong cache, chỉ không vào bảng.
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
    # Đuôi bảng giữ nguyên quy tắc cũ — `since` chỉ cắt ĐẦU bảng, không đụng đuôi.
    end = max(last, max(raw.index), first)
    months = pd.period_range(first, end, freq="M")
    return raw.reindex(months).rename("delta")


def carry_forward(raw: pd.Series):
    """Tháng không có quan sát thì giữ nguyên giá trị của kỳ tính toán gần nhất.

    Trả ``(chuỗi đã bù, chuỗi nguồn)``.  Nguồn là ``observed`` hoặc
    ``carry:<tháng>``.

    Đây là toàn bộ quy tắc vá còn lại.  Bản trước có thêm hai bước vá **trong
    cùng một tháng** — mượn từ rổ ngắn hơn gần nhất, rồi từ rổ gần nhất hai
    chiều — nhưng hai bước đó chỉ tồn tại vì bình quân khi ấy chia theo rổ kỳ
    hạn.  Nay mỗi tháng chỉ có một con số nên trong tháng không còn ô nào để vá:
    hoặc tháng đó có quan sát, hoặc không.

    Không bù lùi: tháng rỗng nằm trước quan sát đầu tiên bị cắt bỏ chứ không
    mượn số của tương lai.
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
    """Chuỗi một ``delta``/tháng -> bảng (tháng × 18 rổ kỳ hạn).

    Mọi cột **giống hệt nhau theo thiết kế**: mô hình chỉ có một ``delta`` cho
    mỗi tháng (xem :func:`monthly_delta_raw`).  Bảng giữ đủ 18 cột vì phía tiêu
    dùng — :func:`delta_for_bond` và ``main_v2.py`` — tra theo rổ kỳ hạn còn lại
    của từng trái phiếu.
    """
    return pd.DataFrame({t: monthly for t in TENOR_ORDER}, index=monthly.index)


# ---------------------------------------------------------------------------
# Hiệu chỉnh cả pool và dựng bảng spread
# ---------------------------------------------------------------------------
CACHE_COLUMNS = ("bond_id", "obs_date", "group", "tenor_bucket", "par_value",
                 "obs_dirty_price", "delta", "status", "n_eval", "resid_price",
                 "message", "price_hash")


def load_cache(path) -> pd.DataFrame:
    """Đọc cache hiệu chỉnh; không có file thì trả khung rỗng đúng cột."""
    import os
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(CACHE_COLUMNS))
    df = pd.read_csv(path, dtype={"price_hash": str})
    missing = [c for c in CACHE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: cache thiếu cột {missing}")
    return df



def _obs_keys(df) -> set:
    """``{(bond_id, 'YYYY-MM-DD')}`` — khoá chung cho cache và file quan sát."""
    if not len(df):
        return set()
    return set(zip(df["bond_id"].astype(str).str.strip(),
                   pd.to_datetime(df["obs_date"]).dt.date.astype(str)))


def _stale_keys(cached, obs) -> set:
    """Khoá có trong cache nhưng không còn trong bộ quan sát."""
    return _obs_keys(cached) - _obs_keys(obs)


def calibrate_all(obs: pd.DataFrame, bond_df, coupon_schedule_df,
                  holiday_calendar, cfg, *, curve_of_group, params_of,
                  ref_df_of=None, base_delta_of=None, cache_path=None,
                  prune_stale=True, verbose=True) -> pd.DataFrame:
    """Dò ``delta`` cho **mọi** quan sát trong ``obs``.

    ``curve_of_group(group) -> zyc_df``, ``params_of(spec) -> ModelParams`` và
    ``ref_df_of(spec) -> ref_df | None`` được truyền vào chứ không tra trong hàm:
    module này không biết gì về thư mục đường cong hay ``specs/hullwhite.json``,
    và nhờ vậy phép kiểm chạy được trên đường cong dựng tay.

    ``base_delta_of(spec, obs_date) -> float`` cấp **nền** cho từng quan sát;
    bỏ trống nghĩa là nền bằng 0 (đường nhóm trần).

    ``params_of`` nhận **spec**, không phải nhóm.  Đường chiết khấu chỉ phụ thuộc
    nhóm, nhưng ``a_L``, ``sigma_L`` và ``rho`` phụ thuộc đường tham chiếu của
    từng mã: truyền nhóm thôi thì mọi mã thả nổi rơi về ``sigma_L = 0`` và cây
    hai nhân tố nổ ngay.

    **Cache theo nội dung, không theo thời điểm.**  Khoá là
    ``(bond_id, obs_date)`` cộng ``price_hash``; đổi bất kỳ đầu vào nào — giá
    quan sát, ``a``, ``sigma``, ``step_days``, hay một ô bất kỳ của đường cong
    nhóm — là hash đổi và dòng đó được dò lại.  Cache theo dấu thời gian file
    sẽ bỏ sót đúng trường hợp nguy hiểm nhất: đường cong bị sửa tại chỗ.

    ``prune_stale`` (mặc định bật) giữ cho tệp cache **luôn mô tả đúng bộ quan
    sát hiện tại**: dòng của ``(bond_id, obs_date)`` không còn trong ``obs`` bị
    xoá.  Trước đây chúng được giữ lại, nên khi thay file quan sát thì dòng của
    bộ dữ liệu cũ sống mãi — tệp cache mô tả một bộ quan sát không còn tồn tại
    và ai đọc nó để dựng lại bảng sẽ ra số khác với bảng thật.  Bảng spread
    không sai vì nó dựng từ ``obs``, nhưng một tệp kết xuất nói dối là đủ để
    phải sửa.

    Đặt ``prune_stale=False`` khi cố ý chạy trên một **tập con** quan sát và
    muốn giữ phần còn lại của cache.
    """
    from src.bond_pricer import BondTermSheet

    cached = load_cache(cache_path)
    # Chỉ nhận lại dòng đã dò THÀNH CÔNG.  Cache một thất bại là tự bịt mắt: lần
    # sau sửa đúng nguyên nhân mà khoá không đổi thì nó vẫn trả về lỗi cũ.
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
                              spec.step_days, zyc_df, ref_df, base_delta=base)
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
        # par_value là của QUAN SÁT, không phải của term sheet — calibrate_delta
        # không nhìn thấy nó nên gán ở đây, kể cả khi lấy từ cache.
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
            # `obs` là toàn bộ bộ quan sát, và `fresh` đã phủ hết phần dò được
            # của nó, nên mọi dòng cũ đều hoặc đã bị thay hoặc đã mồ côi.
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
    """``calib_df`` -> bảng spread hàng tháng.

    Trả ``(spread, zero_eq, provenance)``.  ``a`` chỉ dùng cho bảng phái sinh
    ``zero_eq`` — bảng chính vẫn là ``delta`` trong không gian ``x(t)``.

    ``provenance`` là khung một dòng mỗi tháng, kèm số quan sát và tổng mệnh giá
    đứng sau con số của tháng đó — đọc cùng bảng chính để biết tháng nào là số
    liệu thật và mỏng đến mức nào.
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
# Tra cứu và báo cáo
# ---------------------------------------------------------------------------
def zero_equivalent(spread_df: pd.DataFrame, a: float) -> pd.DataFrame:
    """``delta`` -> spread trên lãi suất zero, ``delta · B_a(tau)/tau``.

    Đây mới là con số đưa vào tài liệu phương pháp luận và báo cáo — ``delta``
    thô không so sánh được giữa các nhóm vì ``a`` khác nhau.
    """
    from callput import hw_B
    tau = np.array([TENOR_MONTHS[t] / 12.0 for t in spread_df.columns])
    return spread_df * (hw_B(a, tau) / tau)


def load_spread_table(path) -> pd.DataFrame:
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
    """Một số ``delta`` cho một trái phiếu tại ngày định giá."""
    m = pd.Period(pd.Timestamp(value_date), freq="M")
    if m not in spread_df.index:
        raise ValueError(
            f"bảng spread không có tháng {m} (có {spread_df.index.min()}.."
            f"{spread_df.index.max()}) — nối dài index khi dựng bảng"
        )
    return float(spread_df.loc[m, remaining_tenor_bucket(value_date, maturity_date)])


def price_hash(obs_dirty_price, group, params, step_days, zyc_df,
               ref_df=None, base_delta: float = 0.0) -> str:
    """Khoá cache theo nội dung: băm **mọi** đầu vào của một lần dò.

    Nguyên tắc là khoá phải phủ hết những gì đổi được kết quả, vì cache sai
    không báo lỗi — nó trả về một con số cũ trông vẫn hợp lý.

    ``a_r`` **bắt buộc** có mặt: nó vừa chi phối động học cây vừa chi phối hình
    dạng vector bump ``B_a(tau)/tau``, nên hiệu chỉnh lại Hull-White cho đường
    nhóm làm mọi ``delta`` cũ vô nghĩa.

    ``base_delta`` cũng phải có: hai pipeline (OAS không tăng vốn và delta tăng
    vốn) chạy trên **cùng** bộ đường cong và **cùng** bộ tham số, chỉ khác nền.
    Bỏ nó khỏi khoá là hai pipeline dùng chung ô cache và trả số của nhau.

    ``a_L``, ``sigma_L``, ``rho`` và cả nội dung đường tham chiếu cũng phải có:
    với mã thả nổi, chân tham chiếu quyết định dòng tiền chứ không chỉ chiết
    khấu.  Khoá chỉ gồm nhóm sẽ coi hai mã cùng nhóm khác đường tham chiếu là
    một.
    """
    h = hashlib.sha1()
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
