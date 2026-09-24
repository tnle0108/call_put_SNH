# Mô hình định giá quyền chọn nhúng trong trái phiếu — CallPut_v3

Nội bộ VCB. Kho này kèm **dữ liệu thật** (term sheet, quan sát giá, đường cong
LSCK/LSTC, kết quả danh mục), không phát hành ra ngoài.

| | |
|---|---|
| Kho | `CallPut_v3`, nhánh `main`, commit `e69ed46`. Lịch sử trước đó ở `call_put_SNH_v2` |
| Quan sát không tăng vốn | **0 dòng — chờ nạp.** Chưa có thì tầng `OAS` = 0 và kết quả giống hệt bản chưa có kiến trúc hai tầng |
| Ngày định giá | 31/03/2026 |
| Phạm vi | 88 mã: 74 trái phiếu tăng vốn (Tier 2), 14 không tăng vốn |
| Được định giá | 51 mã — lọc `ngày phát hành ≤ ngày định giá < ngày đáo hạn` |
| Quan sát giá | 85 dòng, 74 mã, 2017-12 … 2026-08; **23 dòng dùng được** |
| Python | 3.11.7 — numpy 1.26.4, pandas 2.1.4, scipy 1.11.4 |

---

## 1 · Thay đổi so với phiên bản trước

Điểm thay đổi lớn nhất: **bỏ đường cong Tier 2 dựng tay trong Excel**. Trước
đây trái phiếu tăng vốn được chiết khấu trên một đường cong riêng, dựng bằng
cách đắp margin lên báo giá VBMA. Nay mọi trái phiếu — tăng vốn hay không —
đều chiết khấu trên **đường cong LSCK của chính phân nhóm tổ chức phát hành**,
và phần bù được đưa vào dưới dạng lượng dịch trên **biến trạng thái `x(t)`** của
cây, dò ngược từ giá giao dịch quan sát được.

**Có hai tầng phần bù, cộng dồn.** Đường cong nhóm được dựng từ trái phiếu
**không quyền chọn**, trong khi mọi mã trong phạm vi mô hình đều **có** quyền
chọn, nên còn một khoảng cách nữa phải đo:

| Tầng | Ký hiệu | Đo cái gì | Nền |
|---|---|---|---|
| 1 | `OAS` | chênh của trái phiếu **không tăng vốn có quyền chọn** so với đường cong nhóm | đường cong nhóm |
| 2 | `δ` | phần chồng thêm của trái phiếu **tăng vốn có quyền chọn** | đường cong nhóm + `OAS` |

Định giá dùng bốn tổ hợp — hai chân của cùng một trái phiếu nằm trên **hai cây
khác nhau**:

| Loại | Chân có quyền chọn `V_opt` | Chân không quyền chọn `V₀` |
|---|---|---|
| Không tăng vốn | `OAS` | `0` |
| Tăng vốn | `OAS + δ` | `δ` |

Hàng dưới đúng bằng "cây tăng vốn có quyền chọn **trừ** `OAS`". Chân `V₀` không
mang `OAS` vì đường cong nhóm vốn dựng từ trái phiếu không quyền chọn.

> **Hệ quả cần biết khi đọc kết quả.** Vì hai chân nằm trên hai đường, hiệu
> `diff = full − straight` **không còn là giá trị quyền chọn thuần** — nó gồm cả
> giá trị quyền chọn lẫn phần chênh do `OAS`. Đây là hệ quả có chủ đích, không
> phải sai sót. Kèm theo: đẳng thức phân rã `full = straight + sàn − trần − call
> + put + tương tác` chỉ còn đúng trên một đường, nên đừng ghép `diff` với các
> cấu phần của phép phân rã một đường.

> **Đọc hai bảng cho đúng.** Giá trị trong `tier2_spread_monthly.csv` và
> `nontier2_oas_monthly.csv` là lượng
> dịch trên `x(t)`, **không phải** spread cộng thẳng vào lãi suất zero. Phần bù
> thực tế trên đường cong là `δ · B_a(τ)/τ` và **giảm dần theo kỳ hạn**: với
> `δ = −211 bp` và `a = 0,3303` của kỳ báo cáo hiện tại thì −202,5 điểm cơ bản ở
> kỳ hạn 3 tháng nhưng chỉ −21,3 ở 30 năm. Cộng thẳng `δ` vào một lãi suất zero
> sẽ sai vài chục điểm cơ bản ở đầu dài mà không có cảnh báo nào. Bảng đã quy
> đổi sẵn nằm ở `tier2_spread_zero_equivalent.csv` — dùng bảng đó cho báo cáo.

**Một `δ` cho mỗi tháng, không tách theo rổ kỳ hạn.** Dữ liệu không đỡ nổi mức
chi tiết đó: chỉ ba rổ (5Y, 7Y, 10Y) từng có giao dịch trong số 18 rổ chuẩn, và
phần lớn tháng chỉ có một tới hai quan sát. Cấu trúc kỳ hạn của phần bù vẫn còn
nhưng đến từ mô hình — `δ·B_a(τ)/τ` — chứ không từ việc chia rổ.

**`(a, σ)` lấy từ một đường cơ bản duy nhất** (`FI_ZYC_VND_VBMA_Bond_FI`, hằng
số `BASE_CURVE`), vì đường ZYC của từng nhóm là "đường cơ bản + margin" mà margin
cập nhật không đều, nên hiệu chỉnh trực tiếp trên đường nhóm bắt cả nhiễu của
margin. Phần **mức** lãi suất vẫn khớp vào đúng đường nhóm qua quy nạp tiến
Arrow–Debreu: động học vay của đường cơ bản, mức lấy của đường nhóm.

Ba sửa lỗi độc lập với Tier 2, có tác động tới số liệu và cần được ghi nhận
riêng khi so với kỳ trước:

| Lỗi | Tác động đo được |
|---|---|
| `σ` bị nhân dồn `1,25^k` trong vòng lặp kịch bản — kịch bản 6 dùng 3,81 lần `σ` gốc thay vì 1,25 lần | tới **+460 bp** mệnh giá ở kịch bản 6 |
| Ngày ấn định lãi (fixing) lệch so với ngày đặt lại — nay giả định trùng nhau | neo trên lưới 15 → 11 mốc, giá −9,14 bp |
| Sàn/trần không được áp cho kỳ lãi đã cố định | chưa cắn hôm nay (cột `floor`/`cap` rỗng ở cả 42 dòng) nhưng sẽ cắn khi có dữ liệu |

---

## 2 · Nội dung gói

```
MB-01.TL.XDMH_Call Put SNH_v2.docx     Tài liệu kỹ thuật mô hình (bản chính)
MB-01.TL.XDMH_Call Put SNH_v2.pdf      Bản PDF để đọc nhanh, không sửa
Phuong_phap_luan_spread_trai_phieu_tang_von.html
                                       Phương pháp luận phần bù Tier 2, bản
                                       chi tiết đã duyệt (tham chiếu của Mục 2.3.3)
callput/                               Engine định giá: đường cong, chân lãi suất,
                                       lịch dòng tiền, lưới cây, cây call/put
src/                                   Lớp nối với dữ liệu VCB
  bond_pricer.py                         định giá thuần theo ngày định giá
  tier2_spread.py                        dò δ, bình quân theo tháng, nguyên tắc vá
  tier2_spread_selftest.py               39 phép kiểm hàm thuần
  bond_schedule.py, map_curve.py, shock.py, calc_rho.py, daycount.py
notebooks/pipeline/
  main_v2.py                             chạy định giá cả danh mục
  build_spreads.py                       dựng hai bảng phần bù: OAS trái phiếu
                                         không tăng vốn, rồi delta tăng vốn
quantmr/, Quant_Lib/                   Thư viện quant nội bộ (đường cong,
                                       hiệu chỉnh Hull-White, lịch nghỉ lễ)
datasets/                              Dữ liệu đầu vào (xem Mục 4)
specs/hullwhite.json                   Tham số (a, σ) đã hiệu chỉnh cho từng đường
outputs/                               Kết quả chạy
datasets/curve/_deprecated/            Đường cong Tier 2 dựng tay của phương
                                       pháp cũ, giữ để đối chiếu khi nghiệm thu.
                                       Không tệp mã nguồn nào còn đọc chúng.
requirements.txt                       Thư viện cần cài
```

**Không kèm** `.venv.7z` (78 MB): đó là môi trường ảo đóng gói, cắm cứng vào
đường dẫn của một máy khác nên không dùng lại được. Cài mới theo Mục 3.

---

## 3 · Chạy lại

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Thứ tự chạy bắt buộc** — bước 2 đọc kết xuất của bước 1:

```bat
cd notebooks\pipeline

REM Bước 1 — dựng HAI bảng phần bù từ giá giao dịch quan sát được.
REM Chạy lại khi một trong hai file quan sát có dòng mới, hoặc khi đổi
REM kỳ báo cáo — mốc T1 tính lại theo ngày định giá.
REM   chặng 1: OAS trái phiếu không tăng vốn, nền là đường cong nhóm
REM   chặng 2: delta tăng vốn, nền là đường cong nhóm + OAS
REM Thứ tự hai chặng do chính script bảo đảm.
python build_spreads.py

REM Bước 2 — định giá cả danh mục, 51 mã x 7 kịch bản.
python main_v2.py
```

Đường dẫn import phụ thuộc thư mục làm việc: **phải** `cd` vào
`notebooks\pipeline` trước khi chạy, nếu không import sẽ hỏng.

Hai biến môi trường điều khiển `main_v2.py`:

| Biến | Ý nghĩa |
|---|---|
| `CP_ROWS` | Mặc định `all`. Đặt `CP_ROWS=0,5,41` để thu hẹp khi gỡ lỗi. Hai mã kiểu Mỹ dùng bước 5 ngày và chiếm phần lớn thời gian chạy. |
| `CP_DUMP=<đường dẫn>` | Ghi mọi kết quả dạng hex float. Hai lần chạy so được bằng `==`, không cần dung sai — đây là cách từng thay đổi trong gói này được chứng minh là không làm xê dịch con số không đáng xê dịch. |

Chạy phép kiểm hàm thuần (vài giây, không cần dữ liệu):

```bat
python src\tier2_spread_selftest.py
```

---

## 4 · Dữ liệu đầu vào

| Tệp | Nội dung |
|---|---|
| `datasets/raw/bond placeholder.xlsx` | Term sheet 88 mã: ngày, giá thực hiện, biên độ, sàn/trần, kiểu quyền chọn, `group`, `is_tier2` |
| `datasets/raw/tier2_price_obs.csv` | **Quan sát giá dirty dạng dài** cho trái phiếu **tăng vốn** — `bond_id, obs_date, dirty_amount, par_value, coupon_rate` |
| `datasets/raw/nontier2_price_obs.csv` | Cùng lược đồ, cho trái phiếu **không tăng vốn có quyền chọn**. Sinh ra tầng `OAS`. Hai pool tách bạch: nạp nhầm mã sang pool kia thì mô hình dừng và báo lỗi |
| `datasets/curve/LB_G{1,2,3}.csv` | Báo giá YTM theo phân nhóm tổ chức phát hành |
| `datasets/curve/FI_ZYC_VND_LB_G{1,2,3}.csv` | Đường zero đã bootstrap (bộ nhớ đệm) |
| `datasets/curve/SOB4.csv` | Đường lãi suất tham chiếu cho mã thả nổi |
| `datasets/correlation/corr.csv` | Tương quan `ρ` giữa hai nhân tố |
| `specs/hullwhite.json` | `(a, σ)` đã hiệu chỉnh Kalman cho từng đường |

> ### ⚠ Nguồn giá: phát hành sơ cấp, và rất mỏng
>
> Cả 85 quan sát trong `tier2_price_obs.csv` đều có **giá dirty đúng bằng 100%
> mệnh giá**. Đây là dữ liệu **phát hành sơ cấp** (phát hành ngang mệnh giá),
> không phải giao dịch thứ cấp — đây là nguồn giá duy nhất hiện có.
>
> Hệ quả về ý nghĩa: `δ` đo **spread phát hành** — chênh giữa lãi suất coupon ấn
> định lúc phát hành và đường cong của phân nhóm tổ chức phát hành — chứ không
> phải OAS thị trường thứ cấp. Hai đại lượng này không thay thế nhau được.
>
> **Chỉ 23/85 quan sát dùng được.** 62 dòng có ngày trước 18/09/2023, tức trước
> khi lịch sử đường cong nhóm bắt đầu, nên bị loại (`before_curve_history`).
> Phần lớn nằm ở quý 4/2021.
>
> **`δ` âm ở phần lớn các tháng gần đây**, nghĩa là trái phiếu tăng vốn đang
> được định giá chặt hơn đường cong nhóm — ngược với bản chất nợ thứ cấp. Nhiều
> khả năng do lãi suất coupon phát hành sơ cấp được ấn định theo quy trình, trễ
> so với đường cong thứ cấp VBMA. Đã chấp nhận vì không có nguồn giá khác.
>
> **Tháng dùng để định giá (2026-03) là `carry` từ 2025-08**, tức dựa trên **một
> giao dịch duy nhất**. Bảng `tier2_spread_provenance.csv` có hai cột
> `so_quan_sat` và `tong_menh_gia` chính là để chỗ này không bị đọc lướt.
>
> Ba cột trong file quan sát: `dirty_amount` (số tiền thanh toán, đã gồm lãi dự
> thu — dấu là chiều vị thế, mô hình lấy trị tuyệt đối), `par_value` (mệnh giá
> **của chính giao dịch đó**, cũng là trọng số bình quân) và `coupon_rate` (lãi
> suất kỳ coupon hiện hành tại ngày quan sát, dạng thập phân).

---

## 5 · Kết quả

| Tệp | Nội dung |
|---|---|
| `outputs/bond_results_tier2_oas.xlsx` | **Kết quả hiện hành**: 51 mã × 7 kịch bản, các cột `full_price`, `straight_price`, `diff`, `shock` |
| `datasets/spread/tier2_spread_monthly.csv` | Bảng `δ` theo tháng (không gian `x`). Có 18 cột kỳ hạn nhưng **giống hệt nhau theo thiết kế** — mô hình chỉ có một `δ` cho mỗi tháng |
| `datasets/spread/tier2_spread_zero_equivalent.csv` | Bảng quy đổi `δ·B_a(τ)/τ` — dùng cho báo cáo |
| `datasets/spread/tier2_spread_provenance.csv` | Nguồn từng ô: `observed` / `shorter:<rổ>` / `nearest:<rổ>` / `carry:<tháng>` |
| `datasets/spread/tier2_spread_calibrated.csv` | `δ` dò được cho từng quan sát, kèm khoá băm nội dung |
| `datasets/spread/nontier2_oas_monthly.csv` | Bảng `OAS` theo tháng — tầng 1, nền cho tầng 2 |
| `datasets/spread/nontier2_oas_{zero_equivalent,provenance,calibrated}.csv` | Ba file phái sinh cùng dạng với tầng tăng vốn |
| `outputs/bond_results_*.xlsx` (các tệp còn lại) | **Lịch sử**, từ các lần chạy trước. Tên tệp mã hoá nút điều khiển đang được thay đổi lúc đó; nhiều nút trong số đó nay không còn tồn tại. Không dùng để đối chiếu. |

Đọc `tier2_spread_provenance.csv` cùng bảng chính: nó cho biết ô nào là số liệu
quan sát thật và ô nào được vá, tức mức độ tin cậy của từng kỳ hạn không như
nhau.

---

## 6 · Kiểm chứng đã thực hiện

Mỗi thay đổi được chứng minh riêng là chỉ làm xê dịch đúng cái nó phải làm xê
dịch. Phương pháp: chạy trước và sau, kết xuất hex float, so bằng `==`.

| Thay đổi | Kiểm chứng |
|---|---|
| Giải `φ(t)` bằng quy nạp tiến thay cho `A` giải tích | trùng engine gốc tới `4,3 × 10⁻¹²` bp trên cả 1 và 2 nhân tố, bước 21d và 56d, có call/put/floor |
| Tách `src/bond_pricer.py` | kết xuất và nhật ký **trùng từng byte** |
| Dọn code chết, xoá nút điều khiển | 6 mã × 7 kịch bản (gồm cả 2 mã kiểu Mỹ) **trùng từng bit** |
| Xoá cột `premium`, xoá `bootstrap_curve.py` | 5 mã × 7 kịch bản **trùng từng bit** |
| Nối dây spread Tier 2 | 3 mã **không** tăng vốn trùng từng bit với bản chạy không có bảng spread |
| Dò ngược `δ` | trên dữ liệu giả định của bản trước: 7/7 quan sát dò lại **đúng** `δ` đặt trước |
| Bình quân trọng số par | hai quan sát, par 3:1, `δ` 100/200 bp → đúng 125 bp |
| Giữ giá trị kỳ gần nhất | chạy đủ hai nhánh `observed` / `carry`, gồm cả chuỗi nhiều tháng rỗng liên tiếp và tháng rỗng dẫn đầu |
| Hợp đồng ghi cache | thay bộ quan sát thì dòng cũ bị xoá; `prune_stale=False` vẫn gộp thêm |
| Gỡ mã kiểm thử | 51 mã được định giá, **không mã `Test_*` nào** trong kết quả |
| `δ` chỉ chạm trái phiếu tăng vốn | 91 dòng của mã không tăng vốn đổi **đúng 0,0 bp** khi bảng spread thay đổi |
| Đặc tả `R_T2 − R_G = δ·B_a(τ)/τ` | đúng tại 4.000 điểm từ 1 ngày tới 30 năm, sai số `< 0,001` bp |
| **Cổng xếp tầng** | Bật kiến trúc hai tầng khi chưa có dữ liệu `OAS`: 357 dòng kết quả trùng **từng bit** với bản trước, trên cả ba cột |
| **Bất biến của tổng** | `nền + δ` bằng đúng `δ` khi nền `= 0`, lệch `1,0e−05` và `1,4e−06` bp — đúng dung sai Brent. Bài toán chỉ phụ thuộc tổng và nghiệm duy nhất do tính đơn điệu |
| Hai cây cùng lưới | Lưới ngày, `Δx`, mức `x`, xác suất nhánh trùng **từng bit**; chỉ `φ` khác. Sai số rời rạc hoá vẫn triệt tiêu khi lấy hiệu |
| Hàm thuần | `src/tier2_spread_selftest.py` — 57 phép kiểm, `src/tier2_cache_selftest.py` — 9, `src/layered_selftest.py` — 18. Tất cả PASS |

---

## 7 · Hạn chế đã biết

**7.1 · Bước hiệu chỉnh FRA của Hull-White chưa được cài.** Bước khớp drift làm
cây tái tạo đúng *giá trái phiếu không coupon* của đường tham chiếu, nhưng điều
kiện thị trường đúng phải là *lãi suất kỳ hạn* khớp — hai ràng buộc khác nhau.
Đo được **3,92 bp mệnh giá** ở `ρ = 0`, tỷ lệ chính xác với bình phương độ biến
động của đường tham chiếu trên dải kiểm thử rộng 66 lần. Vì phần lồi mang dấu
dương, cây chưa hiệu chỉnh cho lãi suất kỳ hạn kỳ vọng cao hơn thị trường, nên
`δ` dò ra ở các mã thả nổi hơi cao hơn thực tế. Ảnh hưởng 5/25 mã tăng vốn (loại
thả nổi theo SOB4); mã lãi suất cố định không chịu tác động. Chi tiết tại Mục
5.1 của MB-01.

**7.2 · Lịch nghỉ lễ chưa có hiệu lực.** Hai lỗi chồng nhau: `main_v2.py` trỏ
`datasets/holidays` (số nhiều) trong khi thư mục thật là `datasets/holiday` (số
ít) — `load_holiday_calendar` trả về rỗng mà không báo lỗi; và ngay cả khi sửa
đường dẫn thì `vnd.csv` **hết ngày lễ từ 02/09/2025**, trong khi ngày định giá
là 31/03/2026. Đo được: sửa đường dẫn làm đổi **0 trên 432** ngày trả lãi của cả
sổ, nên hiện chưa ảnh hưởng số liệu. Muốn lịch có hiệu lực thật phải bổ sung
ngày lễ từ 2026 (nguồn có sẵn: `Quant_Lib/metadata/HLD/holidays_2026.xlsx`).

**7.3 · Hệ số `λ` của bước hiệu chỉnh tương quan.** Khi lượng hiệu chỉnh
`ρ/36 · M` đẩy một xác suất xuống âm, mô hình nhân hệ số `λ < 1` để giữ xác suất
hợp lệ. Hệ quả: tương quan hiệu dụng tại nút đó **thấp hơn** `ρ` đưa vào. Chỉ xảy
ra ở rìa cây nơi khối lượng xác suất nhỏ. Chi tiết tại Mục 2.3.2 (g) của MB-01.

**7.3b · Nguồn giá quá mỏng để nói về cấu trúc kỳ hạn của phần bù.** Chỉ 23
quan sát dùng được, trải 12 tháng trong ba năm, và tháng dùng để định giá dựa
trên một giao dịch duy nhất. Đây là lý do mô hình chốt một `δ` cho mỗi tháng
thay vì chia theo rổ kỳ hạn. Khi có thêm nguồn giá — nhất là giao dịch thứ
cấp — nên xem lại quyết định này.

**7.3c · `(a, σ)` không hiệu chỉnh riêng cho từng phân nhóm.** Cả ba nhóm dùng
chung tham số của đường cơ bản (`BASE_CURVE`), vì đường nhóm là "đường cơ bản +
margin" và margin cập nhật không đều. Hệ quả: khác biệt giữa các nhóm nằm ở
*mức* đường cong chứ không ở *động học*. Chấp nhận có chủ đích, không phải sót.

**7.3d · `Opt` không còn là giá trị quyền chọn thuần.** Khi `OAS ≠ 0`, hai chân
nằm trên hai đường chiết khấu khác nhau, nên `diff = full − straight` gồm cả giá
trị quyền chọn lẫn phần chênh do `OAS`. Đây là hệ quả có chủ đích của việc chọn
đường cong nhóm — vốn dựng từ trái phiếu không quyền chọn — làm đường cho chân
không quyền chọn. Hệ quả: không ghép `diff` với các cấu phần của `decompose()`,
vốn chỉ đúng trên một đường. Xem MB-01 Mục 2.3.3 (f).

**7.3e · Tầng `OAS` phủ được ít tháng hơn tầng `δ`.** Quan sát trái phiếu không
tăng vốn bắt đầu muộn hơn quan sát trái phiếu tăng vốn, nên mốc `T₁` — điểm bắt
đầu chung của hai bảng — bị kéo về gần kỳ báo cáo và phần lịch sử trước đó rơi
ra ngoài bảng. Các quan sát đó vẫn được dò và vẫn nằm trong tệp `*_calibrated.csv`
để đối chiếu, nhưng `δ` của chúng dò trên nền đường cong nhóm trần nên **không so
sánh được** với `δ` từ `T₁` trở đi. Nạp thêm quan sát trái phiếu không tăng vốn ở
các tháng cũ sẽ đẩy `T₁` lùi lại và mở rộng bảng.

**7.4 · Lãi suất kỳ đầu của trái phiếu thả nổi đang được suy từ đường cong.** Với
kỳ đã ấn định, `build()` suy lãi từ đường tham chiếu thay vì đọc mức đã công bố.
Cột `coupon_rate` trong `tier2_price_obs.csv` đã được điền đủ 85/85 dòng nhưng chưa có đường dẫn nào đưa nó vào định giá — đây là việc còn lại.

**7.5 · Không có bộ kiểm tự động cho engine trong cây này.** 75 phép kiểm của
engine nằm ở kho `vcb-callput-embed`. Thay đổi engine trong gói này đã được đối
chiếu ngược lại bản đó, nhưng chưa chuyển thành `pytest` chạy tại chỗ.

---

## 8 · Nhật ký thay đổi

Kho `CallPut_v3` bắt đầu lịch sử git từ trạng thái này. **Bảng dưới là lịch sử
của kho tiền nhiệm `call_put_SNH_v2`** (nhánh `tn_main_dev_2`) — giữ lại vì đó
là nơi ghi lý do và số liệu kiểm chứng của từng thay đổi engine. Từ đây trở đi
dùng `git log` của chính kho này.

| Commit | Nội dung |
|---|---|
| `b2dfb9b` | README và kết quả chạy đủ danh mục |
| `2d51b92` | MB-01: bổ sung dẫn dắt và chứng minh, thay khối Tier 2 cũ |
| `06b06c8` | Hoàn tất luồng spread tăng vốn, chạy thông với dữ liệu giả định |
| `f7b956c` | Xoá đường cong per-bond sinh từ `BufferYTM` |
| `5355f40` | Xoá cột `premium` và `src/bootstrap_curve.py` |
| `ab6d0f2` | Đưa đường cong Tier 2 dựng tay vào `_deprecated` |
| `c665ec2` | Xoá nhánh đường cong Tier 2 dựng tay và các nút điều khiển đã chết |
| `c656f28` | Quan sát giá tăng vốn chuyển sang file dạng dài |
| `47c7e5d` | Áp sàn/trần cho cả kỳ lãi đã cố định |
| `547ab42` | Di trú term sheet và nối dây spread tăng vốn vào pipeline |
| `f416f4a` | Thêm `src/tier2_spread.py` và selftest |
| `d4f948f` | Tách `src/bond_pricer.py` — định giá thuần theo ngày định giá |
| `8269ce8` | Sửa lỗi `σ` dồn `1,25^k` trong vòng lặp sốc, thêm harness hồi quy |
| `6a9d30c` | Giả định ngày fixing = ngày đặt lại lãi suất |
| `de73cc9` | Giải `φ(t)` bằng quy nạp tiến thay vì `A` giải tích cộng hệ số sửa |
