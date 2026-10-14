
**Ô Chất lượng**

- **Dòng trùng:** 0 dòng trùng hoàn toàn và 0 khóa trùng trong cả 5 file gốc; 0 cặp (chuỗi, tuần) trùng trong bảng dài.
- **Ô trống:** file Sales không có ô trống. File Price trống 71.76% (Phase 0) và 56.24% (Phase 1), và **toàn bộ** là ô "không áp dụng": giá trống khi và chỉ khi tuần đó không bán (1.946.496 ô). Không có ô nào có bán mà thiếu giá, và không có ô nào không bán mà có giá.
- **Giá trị giả dạng thiếu:** không có chuỗi ký tự kiểu 'nan', 'unknown', '-' và không có giá trị âm. Có 5.605 ô giá bằng 0 ở tuần có bán (0.69% số ô có bán), thuộc 1.657 chuỗi; 64 chuỗi có giá bằng 0 ở mọi tuần có bán nên không tính được doanh thu.
- **Mâu thuẫn:** các quy tắc miền giá trị và liên trường đều không có vi phạm, trừ giá bằng 0 nêu trên. Ba proxy về độ chính xác có vi phạm: giá nhảy hơn 10 lần giữa hai tuần liền nhau (354 cặp, 0.053%), cùng sản phẩm cùng tuần mà giá chênh hơn 10 lần giữa các chuỗi (3.357 cặp, 3.08%), đột biến hơn 20 lần trung vị (1.495 ô, 0.18%).
- **Giá trị lặp:** 20.557 ô có bán (2.54%) nằm trong đoạn lặp lại y hệt từ 13 tuần liên tiếp, thuộc 374 chuỗi. Đây là dấu hiệu giá trị được điền chứ không phải doanh số ghi nhận hàng tuần.
- **Sáu chiều (theo ô, trước → sau):** Đầy đủ 1.0000 → 0.9994; Duy nhất 1.0000 → 1.0000; Hợp lệ 0.9984 → 1.0000; Nhất quán 1.0000 → 1.0000; Chính xác (proxy) 0.9723, không đổi.
- **Rò rỉ:** cờ "giá trống" dự đoán `sales > 0` với AUC = 1.0000 trên dữ liệu gốc (s_leak = 0.0000). Không được dùng cờ này hay `price_raw` làm đặc trưng.

**Ô Thời gian**

- **Khoảng dữ liệu:** 2020-07-06 đến 2024-01-01 (183 tuần có số lượng và giá), cộng Phase 2 từ 2024-01-08 đến 2024-04-01 (13 tuần, chỉ có số lượng).
- **Lịch:** mọi cột tuần là thứ Hai, cách nhau đúng 7 ngày, không thiếu tuần nào; ba phase nối liền nhau, không chồng và không hở.
- **Số tuần theo năm (Phase 0+1):** 2020: 26, 2021: 52, 2022: 52, 2023: 52, 2024: 1.
- **Chia tập:** huấn luyện 131 tuần (đến 2023-01-02), kiểm tra 52 tuần (từ 2023-01-09). Trong tập kiểm tra có 4 mốc dự báo 13 tuần không chồng nhau, hoặc 40 mốc nếu trượt từng tuần.
- **Cắt trái:** 2.224 chuỗi đã bán ngay tuần đầu của dữ liệu nên ngày ra mắt thật không biết. 3.727 chuỗi có lần bán đầu tiên nằm trong tập kiểm tra.
- **Kịp thời:** dữ liệu kết thúc 2024-04-01, cách hôm nay (2026-10-13) 132 tuần. Dùng cho nghiên cứu và so sánh phương pháp thì không ảnh hưởng; không dùng được để dự báo vận hành hiện tại.
- **Đơn vị giá:** không rõ loại tiền. Giá trung vị của chuỗi là 30.0; 1% đến 99% số ô nằm trong khoảng 2.16 đến 1,349.
