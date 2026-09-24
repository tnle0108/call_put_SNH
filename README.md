# Mô hình định giá quyền chọn nhúng trong trái phiếu — Sổ Ngân hàng

Định giá trái phiếu có quyền chọn mua/bán và sàn/trần lãi suất trên cây tam
thức Hull-White hai nhân tố, báo cáo giá trị quyền chọn nhúng là
`full − straight` và chênh lệch giá trị kinh tế dưới 6 kịch bản cú sốc lãi suất
theo Thông tư 83/2025/TT-NHNN.

**Đọc `DOC_TRUOC_KHI_CHAY.md` trước** — bản kê nội dung, thứ tự chạy, tình
trạng dữ liệu và hạn chế đã biết.

| | |
|---|---|
| Phương pháp luận | `MB-01.TL.XDMH_Call Put SNH_v2.docx` (Mục 2.3) |
| Phần bù trái phiếu tăng vốn | `Phuong_phap_luan_spread_trai_phieu_tang_von.html` |
| Engine định giá | `callput/` — xem `callput/README.md` |
| Lớp nối dữ liệu VCB | `src/` |

```bat
cd notebooks\pipeline
python build_spreads.py         REM dựng hai bảng phần bù (OAS rồi delta)
python main_v2.py               REM định giá cả danh mục
```

Import phụ thuộc thư mục làm việc: phải `cd` vào `notebooks\pipeline` trước khi
chạy.
