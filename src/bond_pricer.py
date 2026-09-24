"""Một dòng term sheet + một ngày định giá -> CallPutTree.

Tách nguyên trạng từ thân vòng lặp ``notebooks/pipeline/main_v2.py`` (các khối
dòng 53-76, 107-154, 232-245, 262-263, 265-301 của bản trước khi tách).

Điểm cốt lõi: **thuần theo ``rpd``**.  Module không đọc file, không ghi file,
không đọc biến toàn cục; mọi thứ đi qua tham số.  Đổi ``rpd`` là đổi ngày định
giá, không phải sửa gì khác.  Đó là điều kiện để module hiệu chỉnh spread Tier 2
định giá được tại *ngày quan sát giá* thay vì tại ``VALUE_DATE``.

Phần I/O — đọc đường cong, cache ``FI_ZYC_VND_*.csv``, ``CurveNode.get``, đọc
``specs/hullwhite.json``, ``calc_rho``, vòng lặp sốc, ghi Excel — **ở lại**
``main_v2.py``.  Module chỉ nhận DataFrame đường cong đã sẵn sàng.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from callput import (
    CallPutTree, CurveLeg, PricingFlags, YieldCurve, compile_bond, hw_B,
)
from src.bond_schedule import CouponSchedule, adjust_following, build
from src.map_curve import MapCurve

__all__ = [
    "PricingConfig", "BondTermSheet", "ModelParams", "PricingResult",
    "refine_curve", "shift_x", "assert_curve_covers", "selected_fixing",
    "map_curves", "build_schedule", "build_legs", "build_tree", "price_bond",
]


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PricingConfig:
    """Hằng số mô hình — bất biến theo trái phiếu và theo ngày định giá."""

    curve_folder: str
    disc_convention: str = "ACT/365"
    ref_convention: str = "ACT/365"
    norm_step_days: float = 21.0
    ame_step_days: float = 5.0
    min_step_days: float = 3.0
    # 0 nghĩa là ngày fixing = ngày đặt lại lãi suất.  bond_schedule.build dùng
    # `if fixing_lag_days` nên 0.0 (falsy) đi thẳng nhánh không-trừ-ngày.
    fixing_lag_days: float = 0.0
    calendar_country: str = "vnd"


@dataclass(frozen=True)
class ModelParams:
    """Tham số Hull-White đã hiệu chỉnh, cho cả hai nhân tố."""

    a_r: float
    sigma_r: float
    a_L: float = 0.0
    sigma_L: float = 0.0
    rho: float = 0.0

    def with_vol_scale(self, k: float) -> "ModelParams":
        """Nhân cả hai sigma với ``k``, trả về bản mới."""
        return replace(self, sigma_r=k * self.sigma_r, sigma_L=k * self.sigma_L)


# ---------------------------------------------------------------------------
# Term sheet
# ---------------------------------------------------------------------------
@dataclass
class BondTermSheet:
    """Một dòng term sheet cùng lát cắt lịch coupon của nó.

    Tên là ``BondTermSheet`` chứ không phải ``BondSpec``: ``src/__init__.py`` đã
    export sẵn một class tên ``BondSpec`` từ ``src.multi_hw_tree``.

    Mọi thuộc tính ở đây **độc lập với ngày định giá**.  Phần phụ thuộc ngày nằm
    duy nhất ở :meth:`option_frames`.
    """

    frame: pd.DataFrame
    """Lát cắt một dòng của ``bond_df``.  Phải là DataFrame chứ không phải
    Series: :meth:`CouponSchedule.bulid_reset_schedule` đọc nó như một khung, và
    ``row.to_frame().T`` sẽ ép mọi cột về ``object``, đổi cách pandas diễn giải
    ngày tháng."""

    coupon_schedule: pd.DataFrame
    """**Toàn bộ** các kỳ trả lãi của trái phiếu, chỉ lọc theo ``bond_id``.
    Không được lọc theo ngày định giá: ``build()`` lùi lại từ kỳ đầu tiên để dựng
    ``accrual_start`` nên cần cả các kỳ đã qua."""

    holiday_calendar: dict
    cfg: PricingConfig

    row: pd.Series = field(init=False, repr=False)
    bond_id: str = field(init=False)
    issue_date: pd.Timestamp = field(init=False)
    style: str = field(init=False)
    ref_curve_name: "str | None" = field(init=False)
    maturity_date: pd.Timestamp = field(init=False)
    coupon_accrual: float = field(init=False)
    face_value: float = field(init=False)
    group: str = field(init=False)
    fixed_rate: "list | None" = field(init=False, repr=False)
    reset_dates: list = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.frame) != 1:
            raise ValueError(
                f"BondTermSheet cần đúng 1 dòng, nhận {len(self.frame)}"
            )
        row = self.frame.iloc[0]
        self.row = row
        self.bond_id = str(row["bond_id"])
        self.issue_date = pd.to_datetime(row["issue_date"])
        self.style = str(row["style"]).strip().lower()
        self.ref_curve_name = (
            None if pd.isna(row["ref_curve"]) else str(row["ref_curve"]).strip().lower()
        )
        # Bản gốc truyền NGUYÊN dict holiday_calendar chứ không phải
        # holiday_calendar.get(country, set()) như CouponSchedule làm.  Với dict
        # rỗng hai cách cho cùng kết quả; giữ nguyên để không đổi hành vi.
        self.maturity_date = adjust_following(
            pd.to_datetime(row["maturity_date"]), self.holiday_calendar
        )
        self.coupon_accrual = float(row["coupon_accrual"])
        self.face_value = float(row["face"])
        self.group = str(row["group"])

        if self.ref_curve_name is not None:
            self.fixed_rate = None
            self.reset_dates = CouponSchedule(
                df=self.frame,
                holiday_calendar=self.holiday_calendar,
                country=self.cfg.calendar_country,
            ).bulid_reset_schedule()  # [sic] tên sai chính tả trong repo
        else:
            self.fixed_rate = [
                float(x) for x in str(row["annual_coupon_rate"]).split(";")
            ]
            self.reset_dates = []

    # -- dựng ---------------------------------------------------------------
    @classmethod
    def from_bond_df(cls, bond_df, coupon_schedule_df, holiday_calendar, cfg,
                     *, iloc=None, bond_id=None) -> "BondTermSheet":
        if (iloc is None) == (bond_id is None):
            raise ValueError("chỉ định đúng một trong iloc / bond_id")
        frame = (bond_df.iloc[[iloc]] if iloc is not None
                 else bond_df[bond_df["bond_id"] == bond_id])
        key = frame.iloc[0]["bond_id"]
        return cls(
            frame=frame,
            coupon_schedule=coupon_schedule_df[coupon_schedule_df["bond_id"] == key],
            holiday_calendar=holiday_calendar,
            cfg=cfg,
        )

    # -- thuộc tính suy ra --------------------------------------------------
    @property
    def is_floating(self) -> bool:
        return self.ref_curve_name is not None

    @property
    def is_american(self) -> bool:
        return self.style == "american"

    @property
    def step_days(self) -> float:
        return self.cfg.ame_step_days if self.is_american else self.cfg.norm_step_days

    # -- lịch quyền chọn ----------------------------------------------------
    def option_frames(self, rpd, *, anchor=None):
        """``(call_df, put_df)`` tại ngày định giá ``rpd``.

        Thang quyền chọn kiểu American neo vào ``max(issue_date, anchor)``, mà
        mặc định ``anchor = rpd`` — tái lập đúng ``max(issue_date, VALUE_DATE)``
        của bản cũ.  Hệ quả: đổi ``rpd`` một ngày làm **toàn bộ** thang dịch một
        ngày, nên lưới đổi và giá lệch một lượng cỡ sàn nhiễu rời rạc hoá.  Module
        hiệu chỉnh nên truyền ``anchor=spec.issue_date`` để thang cố định.
        """
        rpd = pd.Timestamp(rpd)
        anchor = rpd if anchor is None else pd.Timestamp(anchor)
        row, face = self.row, self.face_value
        freq = f"{str(self.cfg.ame_step_days)}D"

        def ladder(strike_col):
            dates = pd.date_range(
                start=max(self.issue_date, anchor),
                end=self.maturity_date,
                freq=freq,
            ).tolist()
            return dates, [float(row[strike_col]) / face] * len(dates)

        def explicit(date_col, strike_col):
            dates = [pd.to_datetime(x.strip(), format="mixed")
                     for x in str(row[date_col]).split(";")]
            return dates, [float(x) / face for x in str(row[strike_col]).split(";")]

        if (self.is_american and pd.notna(row["call_strike"])
                and float(row["call_strike"]) > 0):
            call_dates, call_strikes = ladder("call_strike")
        elif pd.notna(row["call_exercise_dates"]) and str(row["call_exercise_dates"]).strip():
            call_dates, call_strikes = explicit("call_exercise_dates", "call_strike")
        else:
            call_dates, call_strikes = [], []

        if (self.is_american and pd.notna(row["put_strike"])
                and float(row["put_strike"]) > 0):
            put_dates, put_strikes = ladder("put_strike")
        elif pd.notna(row["put_exercise_dates"]) and str(row["put_exercise_dates"]).strip():
            put_dates, put_strikes = explicit("put_exercise_dates", "put_strike")
        else:
            put_dates, put_strikes = [], []

        call_df = pd.DataFrame({
            "call_date": pd.Series(call_dates, dtype="datetime64[ns]"),
            "call_strike": pd.Series(call_strikes, dtype="float64"),
        })
        put_df = pd.DataFrame({
            "put_date": pd.Series(put_dates, dtype="datetime64[ns]"),
            "put_strike": pd.Series(put_strikes, dtype="float64"),
        })
        return call_df, put_df


# ---------------------------------------------------------------------------
# Đường cong: làm mịn lưới và dịch trạng thái x
# ---------------------------------------------------------------------------
_FINE_MAX_YEARS = 30.0
_FINE_POINTS = 1500


def refine_curve(curve: YieldCurve, *, max_years=_FINE_MAX_YEARS,
                 n_points=_FINE_POINTS, tol=1e-9) -> YieldCurve:
    """Cùng một đường cong, trên lưới dày hơn — bảo toàn giá trị tới 1 ULP.

    ``YieldCurve`` nội suy tuyến tính trong không gian lãi suất zero và ngoại suy
    phẳng.  Nội suy tuyến tính một hàm tuyến tính từng khúc trên lưới *chứa* mọi
    nút gốc cho lại đúng hàm đó về mặt toán học.

    Đo được: tại chính các pillar gốc kết quả trùng khớp tuyệt đối; giữa các
    pillar lệch đúng **1 ULP** (1,4e−17 tuyệt đối, 2,2e−16 tương đối) do
    ``np.interp`` làm tròn khác khi đi qua điểm lưới trung gian.  Đây là sai số
    nhỏ nhất có thể của phép làm mịn, không phải chỗ nới lỏng được.
    """
    mats = np.asarray(curve.maturities, float)
    fine = np.geomspace(1.0 / 365.0, max_years, n_points)
    fine = fine[np.min(np.abs(fine[:, None] - mats[None, :]), axis=1) > tol]
    grid = np.unique(np.concatenate([mats, fine]))
    return YieldCurve.from_zero_rates(grid, curve.zero_rate(grid))


def shift_x(curve: YieldCurve, a: float, delta: float, *, refine=True) -> YieldCurve:
    """Dịch nhân tố Hull-White ``x`` đi ``delta``, thể hiện trên đường cong::

        P(0,T)  ->  P(0,T) · exp(−B_a(T)·delta)
        R(T)    ->  R(T) + delta · B_a(T)/T

    Phần bù **không phẳng**: hệ số truyền ``B_a(T)/T`` bằng 1 ở đầu ngắn và tiến
    về ``1/(aT)`` ở đầu dài.

    Lưới mịn là **bắt buộc**, không phải tinh chỉnh: ``B_a(T)/T`` là hàm cong còn
    ``YieldCurve`` nội suy tuyến tính, nên áp phần bù chỉ tại 18 pillar rồi để
    ``np.interp`` lo phần còn lại sẽ lệch tới vài bp giữa các pillar thưa ở đầu
    dài.  Khi đó phép hiệu chỉnh spread hội tụ về sai số nội suy chứ không về giá
    thị trường.

    ``delta == 0`` trả về **chính** object cũ, không đụng gì — nhờ vậy mọi đường
    đi hiện có giữ nguyên từng bit.
    """
    if not delta:
        return curve
    c = refine_curve(curve) if refine else curve
    return c.shifted(delta * hw_B(a, c.maturities) / c.maturities)


# ---------------------------------------------------------------------------
# Ánh xạ đường cong tại một ngày, kèm chặn fallback im lặng
# ---------------------------------------------------------------------------
def assert_curve_covers(rpd, df, label: str) -> None:
    """Nổ nếu ``rpd`` sớm hơn mọi dòng của ``df``.

    ``MapCurve.map_curve`` tụt về ``df.index.min()`` mà **không báo gì** khi không
    có dòng nào ``<= rpd``.  Ở ``VALUE_DATE`` điều đó không bao giờ xảy ra; ở ngày
    quan sát giá trong quá khứ thì có, và nó biến "không có dữ liệu" thành một con
    số trông rất hợp lý.  Chặn ở đây thay vì sửa ``MapCurve``.
    """
    start = pd.Timestamp(pd.DatetimeIndex(df.index).min())
    if pd.Timestamp(rpd) < start:
        raise ValueError(
            f"{label}: rpd={pd.Timestamp(rpd).date()} sớm hơn dòng đầu tiên "
            f"{start.date()} — MapCurve sẽ âm thầm dùng {start.date()}."
        )


def selected_fixing(spec: BondTermSheet, rpd, *, reading="advance"):
    """Ngày fixing mà ``build()`` sẽ chọn cho kỳ coupon tương lai đầu tiên.

    Nhân bản logic ở ``bond_schedule.build`` để chặn trước được; nếu hàm đó đổi
    thì bản sao này phải đổi theo.  Trả ``None`` khi không áp dụng.
    """
    if not spec.is_floating:
        return None
    rpd = pd.Timestamp(rpd)
    pays = sorted(p for p in spec.coupon_schedule["pay_date"] if p > rpd)
    if not pays:
        return None
    pay = pays[0]
    cands = [r for r in spec.reset_dates
             if (r < pay if reading == "advance" else r <= pay)]
    if not cands:
        return None
    sel = max(cands)
    lag = spec.cfg.fixing_lag_days
    return sel - pd.Timedelta(days=lag) if lag else sel


def map_curves(spec, rpd, disc_df, ref_df, *, require_curve_history=False):
    """DataFrame đường cong -> ``(disc_curve, ref_curve)`` tại ``rpd``."""
    if require_curve_history:
        assert_curve_covers(rpd, disc_df, "đường chiết khấu")
        if ref_df is not None:
            assert_curve_covers(rpd, ref_df, "đường tham chiếu")
            fix = selected_fixing(spec, rpd)
            if fix is not None:
                # get_known_rate map lại đường tham chiếu tại ngày fixing, vốn
                # sớm hơn rpd — đây là chỗ fallback dễ kích hoạt nhất.
                assert_curve_covers(fix, ref_df, "đường tham chiếu @fixing")
    cfg = spec.cfg
    disc_curve = MapCurve(
        rpd=rpd, df=disc_df, convention=cfg.disc_convention
    ).map_curve()
    ref_curve = (
        MapCurve(rpd=rpd, df=ref_df, convention=cfg.ref_convention).map_curve()
        if ref_df is not None else None
    )
    return disc_curve, ref_curve


# ---------------------------------------------------------------------------
# Dựng lịch, dựng chân, dựng cây
# ---------------------------------------------------------------------------
def build_schedule(spec, rpd, *, reading="advance", apply_floor=True,
                   apply_cap=True, ref_df=None, call_df=None, put_df=None,
                   option_anchor=None):
    """Thay closure ``build_sched`` của bản cũ; mọi biến nâng lên tham số."""
    if call_df is None or put_df is None:
        c, p = spec.option_frames(rpd, anchor=option_anchor)
        call_df = c if call_df is None else call_df
        put_df = p if put_df is None else put_df
    return build(
        reading=reading,
        rpd=rpd,
        face=spec.face_value,
        maturity_date=spec.maturity_date,
        coupon_accrual=spec.coupon_accrual,
        coupon_schedule_df=spec.coupon_schedule,
        ref_dates=spec.reset_dates,
        call_df=call_df,
        put_df=put_df,
        fixed_rate=spec.fixed_rate,
        fixing_lag_days=spec.cfg.fixing_lag_days,
        ref_curve_name=spec.ref_curve_name,
        curve_folder=str(spec.cfg.curve_folder),
        ref_convention=spec.cfg.ref_convention,
        apply_floor=apply_floor,
        apply_cap=apply_cap,
        ref_df=ref_df,
    )


def _align_bump(src_curve, bump, dst_curve):
    """Bump vô hướng đi thẳng; bump theo pillar phải nội suy sang lưới đích."""
    if np.isscalar(bump):
        return float(bump)
    b = np.asarray(bump, float)
    if b.shape == dst_curve.maturities.shape:
        return b
    return np.interp(dst_curve.maturities, src_curve.maturities, b)


def build_legs(disc_curve, ref_curve, params, cfg, *, disc_bump=0.0,
               ref_bump=0.0, disc_delta=0.0, ref_delta=0.0, refine=True):
    """Thay hàm ``legs`` của bản cũ, thêm đường dịch trong không gian ``x``.

    ``*_bump``  — cộng thẳng vào lãi suất zero (vô hướng hoặc mảng theo pillar).
    ``*_delta`` — dịch nhân tố ``x``, sinh ra phần bù cong ``δ·B_a(τ)/τ``.

    Hai thứ **khác đơn vị lẫn hình dạng**; truyền nhầm sẽ ra kết quả sai mà không
    có lỗi nào báo.  Vector bump được dựng ở đây chứ không ở nơi gọi, vì chỉ ở đây
    mới biết chắc ``a_r`` đi với đường chiết khấu còn ``a_L`` đi với đường tham
    chiếu.

    Với ``disc_delta = ref_delta = 0`` — mọi lời gọi của pipeline hiện tại — hàm
    này thực hiện đúng hai phép ``.shifted()`` y như bản cũ.
    """
    disc_base = shift_x(disc_curve, params.a_r, disc_delta, refine=refine)
    disc = disc_base.shifted(_align_bump(disc_curve, disc_bump, disc_base))
    disc_leg = CurveLeg(disc, params.a_r, params.sigma_r, cfg.disc_convention)
    if ref_curve is None:
        return disc_leg, None
    ref_base = shift_x(ref_curve, params.a_L, ref_delta, refine=refine)
    ref = ref_base.shifted(_align_bump(ref_curve, ref_bump, ref_base))
    return disc_leg, CurveLeg(ref, params.a_L, params.sigma_L, cfg.ref_convention)


def build_tree(spec, rpd, *, disc_df, params, ref_df=None, reading="advance",
               apply_floor=True, apply_cap=True, step_days=None, min_step=None,
               option_anchor=None, disc_bump=0.0, ref_bump=0.0, disc_delta=0.0,
               ref_delta=0.0, refine_on_delta=True,
               require_curve_history=False, sched=None):
    """Một dòng term sheet + một ngày ``rpd`` -> ``(CompiledBond, CallPutTree)``.

    Thứ tự thao tác giữ đúng bản cũ — map đường cong, dựng lịch, compile, dựng
    chân — nên thứ tự các dòng cảnh báo in ra cũng không đổi.
    """
    cfg = spec.cfg
    disc_curve, ref_curve = map_curves(
        spec, rpd, disc_df, ref_df, require_curve_history=require_curve_history
    )
    if sched is None:
        sched = build_schedule(
            spec, rpd, reading=reading, apply_floor=apply_floor,
            apply_cap=apply_cap, ref_df=ref_df, option_anchor=option_anchor,
        )
    bond = compile_bond(
        sched,
        step_days=spec.step_days if step_days is None else step_days,
        min_step=cfg.min_step_days if min_step is None else min_step,
    )
    disc_leg, ref_leg = build_legs(
        disc_curve, ref_curve, params, cfg,
        disc_bump=disc_bump, ref_bump=ref_bump,
        disc_delta=disc_delta, ref_delta=ref_delta, refine=refine_on_delta,
    )
    if ref_leg is None:
        return bond, CallPutTree.single_curve(bond, disc_leg)
    return bond, CallPutTree.multi_curve(bond, disc_leg, ref_leg, rho=params.rho)


@dataclass(frozen=True)
class PricingResult:
    bond_id: str
    rpd: pd.Timestamp
    full_price: float
    straight_price: float
    diff: float
    bond: object = field(repr=False, default=None)
    tree: object = field(repr=False, default=None)


def price_bond(spec, rpd, **kw) -> PricingResult:
    """Định giá đầy đủ: giá có quyền chọn, giá straight, và chênh lệch.

    Giữ nguyên hai lời gọi riêng ``price()`` và ``decompose()`` như bản cũ.
    ``decompose()`` tự tính lại ``full`` nên về mặt số học có thể gộp, nhưng gộp
    là một thay đổi cần chứng minh còn giữ nguyên thì không.
    """
    bond, tree = build_tree(spec, rpd, **kw)
    full = tree.price()
    straight = tree.decompose()["straight"]
    return PricingResult(
        spec.bond_id, pd.Timestamp(rpd), full, straight, full - straight, bond, tree
    )


def price_bond_layered(spec, rpd, *, delta_full, delta_straight,
                       **kw) -> PricingResult:
    """Giá có quyền chọn và giá straight trên **hai đường chiết khấu khác nhau**.

    ``price_bond`` lấy cả hai số từ một cây, nên cả hai nằm trên cùng một đường.
    Ở đây hai chân cố tình tách:

    ====================  ==================  ======================
    Loại                  ``delta_full``      ``delta_straight``
    ====================  ==================  ======================
    Không tăng vốn        ``OAS``             ``0``
    Tăng vốn              ``OAS + delta``     ``delta``
    ====================  ==================  ======================

    Đường `LB_G*` dựng từ trái phiếu **không quyền chọn**, nên nó đúng là đường
    cho chân straight; chân có quyền chọn phải cộng thêm ``OAS`` để khớp giá thị
    trường của chính trái phiếu có quyền chọn đó.  Hàng dưới đúng bằng "cây tăng
    vốn có quyền chọn **trừ** ``OAS``".

    **Hệ quả cần biết khi đọc kết quả**: ``diff = full - straight`` nay gồm cả
    giá trị quyền chọn **lẫn** phần chênh do ``OAS``, không còn là giá trị quyền
    chọn thuần.  Đẳng thức phân rã của ``decompose()`` chỉ còn đúng trên một
    đường, nên đừng đem ``diff`` ở đây ghép với các cấu phần của ``decompose()``.

    Hai cây dùng chung hình học lưới: :class:`FactorLattice` chỉ đọc ``a``,
    ``sigma`` và lưới ngày, **không đọc đường cong**.  Nhờ vậy sai số rời rạc hoá
    vẫn triệt tiêu khi lấy hiệu, đúng như khi cả hai chân ở trên một cây.
    """
    if "disc_delta" in kw:
        raise TypeError(
            "price_bond_layered nhận delta_full / delta_straight, không nhận "
            "disc_delta — truyền nhầm sẽ cho hai chân cùng một đường mà không báo"
        )
    d_full, d_straight = float(delta_full), float(delta_straight)

    bond, tree = build_tree(spec, rpd, disc_delta=d_full, **kw)
    full = tree.price()

    if d_full == d_straight:
        # Một đường -> một cây. Nhánh này cho ra ĐÚNG hai con số của
        # `price_bond`, vì `decompose()["straight"]` chính là
        # `price(PricingFlags.none())`. Đó là cổng hồi quy: khi chưa có OAS,
        # kết quả phải trùng bản cũ từng bit.
        straight = tree.price(PricingFlags.none())
    else:
        bond_s, tree_s = build_tree(spec, rpd, disc_delta=d_straight, **kw)
        if not np.array_equal(bond.days, bond_s.days):
            raise AssertionError(
                "hai cây không cùng lưới ngày — hiệu full - straight sẽ lẫn sai "
                "số rời rạc hoá thay vì chỉ còn giá trị quyền chọn"
            )
        straight = tree_s.price(PricingFlags.none())

    return PricingResult(
        spec.bond_id, pd.Timestamp(rpd), full, straight, full - straight, bond, tree
    )
