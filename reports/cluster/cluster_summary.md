# Tóm tắt phân nhóm kiểu nhu cầu (VN1)

Mốc 1 dùng dữ liệu đến 2023-01-02. Chuỗi đủ điều kiện: 6954; new: 6318; dormant: 1777; no_price: 4.

## Chọn k

| k | Silhouette K-means | Davies-Bouldin K-means | Cụm nhỏ nhất | Silhouette Ward | ARI K-means/Ward |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.238 | 1.82 | 13,7% | 0.147 | 0.41 |
| 4 | 0.169 | 1.78 | 9,3% | 0.165 | 0.42 |
| 5 | 0.176 | 1.60 | 6,5% | 0.158 | 0.42 |
| 6 | 0.182 | 1.55 | 2,3% | 0.124 | 0.45 |
| 7 | 0.183 | 1.51 | 2,2% | 0.125 | 0.41 |
| 8 | 0.168 | 1.69 | 2,2% | 0.124 | 0.35 |

Chọn k = 5. Silhouette của nhãn Syntetos-Boylan trong cùng không gian đặc trưng: 0.075.

## Cụm K-means tại mốc 1

| Cụm | Số chuỗi | % chuỗi | % doanh thu 52 tuần cuối | % tuần bằng 0 | ADI | CV² | Doanh thu tuần | Giá | Tỷ trọng Q4 | Xu hướng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Chủ lực bán đều | 1929 | 27,7% | 73,7% | 3,2% | 1.03 | 0.43 | 637 | 26.1 | 25,2% | +0.06 |
| B2 Sản phẩm giá cao | 454 | 6,5% | 19,2% | 29,9% | 1.43 | 0.63 | 1536 | 346.9 | 17,9% | -0.06 |
| B3 Mùa vụ cuối năm | 618 | 8,9% | 4,6% | 44,7% | 1.81 | 1.69 | 163 | 22.2 | 67,7% | -0.09 |
| B4 Bán ngắt quãng, quy mô nhỏ | 2147 | 30,9% | 2,2% | 41,9% | 1.72 | 0.51 | 45 | 27.1 | 22,9% | -0.18 |
| B5 Bán thưa, lượng nhỏ | 1806 | 26,0% | 0,3% | 84,6% | 6.50 | 0.13 | 5 | 30.0 | 23,8% | -0.29 |

## Syntetos-Boylan tại mốc 1

| Nhóm | Số chuỗi | % chuỗi | % doanh thu 52 tuần cuối |
|---|---:|---:|---:|
| smooth | 1383 | 19,9% | 50,1% |
| erratic | 1286 | 18,5% | 36,1% |
| intermittent | 2559 | 36,8% | 4,9% |
| lumpy | 1726 | 24,8% | 8,8% |

ARI giữa hai cách chia: 0.29.

## Độ ổn định giữa các mốc

| Cặp mốc | Chuỗi đủ điều kiện ở cả hai | % đổi nhãn A | % đổi nhãn B | ARI A | ARI B |
|---|---:|---:|---:|---:|---:|
| 1->2 | 6593 | 8,3% | 12,7% | 0.80 | 0.70 |
| 2->3 | 7735 | 9,1% | 11,7% | 0.78 | 0.73 |
| 3->4 | 8472 | 7,8% | 11,0% | 0.81 | 0.74 |
