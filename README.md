# Dự báo doanh thu 13 tuần tới theo nhóm kiểu nhu cầu cho sản phẩm của nhà bán hàng trực tuyến (dữ liệu VN1)

Đồ án môn DS317 (Khai phá dữ liệu trong doanh nghiệp).

Với mỗi chuỗi nhà bán hàng × kho × sản phẩm của bộ dữ liệu VN1 (15.053 chuỗi, 183 tuần), dự báo doanh thu từng tuần trong 13 tuần kế tiếp. Chuỗi được chia nhóm theo kiểu nhu cầu (Syntetos-Boylan); mô hình gradient boosting được trộn với các phương pháp đơn giản theo tỷ lệ riêng cho từng nhóm. Đánh giá bằng điểm VN1 = MAE% + |Bias%|, tính cho từng kỳ 13 tuần trên bốn kỳ của năm 2023.

## Kết quả chính

| Phương án | Điểm VN1 (doanh thu, trung bình 4 kỳ 2023) |
| --- | --- |
| Mô hình cuối cùng (trộn theo nhóm) | 55,6 |
| Bản đầu | 56,9 |
| Naive | 65,4 |

Trên bài toán số lượng của chính cuộc thi VN1 (Phase 2), phương pháp cho điểm 45,6–47,0% (naive: 50,70%).

## Dữ liệu

Không kèm trong kho này. Tải về bằng Kaggle CLI vào thư mục `raw/`:

```
python -m kaggle datasets download santoshkumarpuvvada/vn1-forecasting-competition-data-set -p raw --unzip
```

Nguồn: bộ dữ liệu Kaggle `santoshkumarpuvvada/vn1-forecasting-competition-data-set` (CC BY 4.0).

## Chạy lại

Cần Python với pandas, numpy, scikit-learn, matplotlib, pyarrow. Chạy theo thứ tự:

| Bước | Lệnh | Kết quả |
| --- | --- | --- |
| Làm sạch | `python scripts/01_clean.py` | `clean/` |
| Khám phá dữ liệu | `python scripts/02_eda.py` | `reports/eda/` |
| Phân nhóm | `python scripts/03_cluster.py` | `clean/clusters/`, `reports/cluster/` |
| Chất lượng dữ liệu | `python scripts/run_notebook.py notebooks/04_chat_luong_du_lieu.py` | `notebooks/04_chat_luong_du_lieu.ipynb`, `reports/dq/` |
| Nhà bán hàng 0 | `python scripts/05_nha_ban_hang_0.py` | `reports/dq/` |
| Bản đầu của mô hình | `python scripts/06_forecast.py` | `reports/forecast/` |
| Mô hình cuối cùng | `python scripts/07_forecast_v2.py` | `reports/forecast_v2/` |
| Cấp tổng hợp và bài toán cuộc thi | `python scripts/08_danh_gia_bo_sung.py` | `reports/danh_gia_bo_sung/` |
| Phân tích sai số | `python scripts/09_phan_tich_sai_so.py` | `reports/phan_tich_sai_so/` |

Các file `.parquet` (dữ liệu sạch và dự báo từng chuỗi) không được đưa lên vì dung lượng; chạy lại các script trên để tạo chúng.

## Thư mục

- `scripts/`: toàn bộ mã nguồn.
- `notebooks/`: notebook kiểm tra chất lượng dữ liệu (đã chạy, kèm kết quả).
- `reports/`: hình, số liệu (`results.json`) và các trang báo cáo HTML của từng giai đoạn. `reports/_archive/` là thử nghiệm cũ đã được thay thế.
