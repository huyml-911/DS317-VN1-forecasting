
**Đề tài:** Dự báo doanh thu 13 tuần tới theo nhóm kiểu nhu cầu cho sản phẩm của nhà bán hàng trực tuyến (dữ liệu VN1)

**Bài toán:** Với mỗi chuỗi nhà bán hàng × kho × sản phẩm, dùng lịch sử bán hàng theo tuần đến thời điểm dự báo để dự báo doanh thu của từng tuần trong 13 tuần kế tiếp; chuỗi được chia nhóm theo kiểu nhu cầu và mỗi nhóm dùng phương pháp phù hợp; đánh giá bằng điểm VN1 (MAE% + |Bias%|) trên bốn kỳ 13 tuần của 52 tuần cuối. Huấn luyện 131 tuần, kiểm tra 52 tuần.

**Kết luận: VN1 là bộ dữ liệu tốt và phù hợp với bài toán này, với điều kiện ghi rõ bốn giới hạn bên dưới.**

| Yêu cầu của bài toán | Dữ liệu đáp ứng thế nào | Đánh giá |
|---|---|---|
| Chuỗi thời gian tuần liên tục, không hở | 196 tuần liên tục, mọi cột là thứ Hai, không thiếu tuần nào | Đạt |
| Mỗi chuỗi được định danh duy nhất | 0 khóa trùng trong 5 file; 15.053 chuỗi có mặt ở cả 5 file | Đạt |
| Giá trị hợp lệ | Không có số âm, không có số lượng lẻ; 0.69% ô có bán có giá bằng 0, đã xử lý | Đạt |
| Tính được doanh thu | Có giá ở mọi tuần có bán; 64 chuỗi không có giá hợp lệ phải loại khỏi phần doanh thu | Đạt, có ngoại lệ nhỏ |
| Đủ dài để chia tập và trượt mốc | 183 tuần; 4 mốc 13 tuần không chồng nhau, hoặc 40 mốc trượt | Đạt |
| Đủ đa dạng để phân nhóm kiểu nhu cầu | Có đủ cả bốn nhóm Syntetos-Boylan với số lượng lớn | Đạt |
| Không rò rỉ từ đặc trưng sang mục tiêu | Giá gốc rò rỉ hoàn toàn (AUC = 1.00); cột giá đã điền thì không (AUC = 0.50) | Đạt sau làm sạch |
| Độ chính xác so với thực tế | Không có nguồn tham chiếu; proxy = 0.9723 | Không kiểm chứng được |
| Học được mùa vụ ở từng chuỗi | Huấn luyện chỉ chứa ba mùa cuối năm, nhiều chuỗi ra mắt muộn | Hạn chế |

**Bốn giới hạn cần ghi trong báo cáo**

1. **Dự báo doanh số đã ghi nhận, không phải nhu cầu.** Không có dữ liệu tồn kho nên tuần bằng 0 có thể là hết hàng hoặc không ai mua.
2. **Chiều Chính xác chỉ đo bằng proxy.** "Chất lượng tốt" ở đây nghĩa là nhất quán nội tại, chưa được đối chiếu với nguồn bên ngoài.
3. **Nhà bán hàng 0** (930 chuỗi, 32% doanh thu huấn luyện, 21% doanh thu kiểm tra) có số lượng là bội số của 7 ở 75% số ô có bán trong giai đoạn huấn luyện, so với 10% trong giai đoạn kiểm tra. Nên báo cáo sai số dự báo tách riêng nhà bán hàng này.
4. **Chuỗi mới và chuỗi ngắn.** 3.727 chuỗi có lần bán đầu tiên nằm trong giai đoạn kiểm tra, nên phải đánh giá bằng mốc trượt và có phương án riêng cho chuỗi thiếu lịch sử.

Không giới hạn nào trong bốn điều trên làm bài toán không thực hiện được; chúng quyết định cách đánh giá và cách diễn giải kết quả.
