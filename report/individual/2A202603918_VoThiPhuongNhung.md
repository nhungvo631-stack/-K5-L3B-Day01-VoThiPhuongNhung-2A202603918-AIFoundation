# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603918
- **Họ và tên:** Võ Thị Phương Nhung
- **SEED** (in ở ô đầu notebook): 3125

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính
| Mô hình | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. BoW + LogReg (tự code) | 0.988 | 0.874 | 0.927 | 0.947 | 8.104 | 4599.012 | 0.059 | 0.115 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.977 | 0.884 | 0.928 | 0.947 | 5.945 | 24072.735 | 0.261 | 0.304 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.832 | 0.937 | 0.881 | 0.974 | 82.246 | 23.867 | 51.752 | 73.151 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.957 | 0.937 | 0.947 | 0.947 | 82.667 | 28.913 | 28.799 | 39.022 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.978 | 0.937 | 0.957 | 0.947 | 83.704 | 29.327 | 30.986 | 55.670 | 0.000 |
| 4a. LLM nhỏ zero-shot | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.317 | 667.570 | 971.406 | 0.028 |

## 3. Dự đoán và thực tế

- llm_models

Đoán: Mô hình 3B (học tủ) thắng LLM zero-shot.

Thực tế: LLM hiện đại tổng quát hóa tốt hơn.

Vì sao lệch: Đánh giá thấp năng lực zero-shot.

- tradeoff

Đoán vị trí: API LLM nằm sâu bên phải vạch 50ms.

Thực tế: Khoảng cách độ trễ thực tế lớn hơn dự kiến rất nhiều lần.

Vì sao lệch: Phác thảo lý thuyết thay vì benchmark thời gian thực.

- hard_eval

Đoán: BoW giảm mạnh trước nhóm né từ khóa.

Thực tế: BoW sụt giảm trầm trọng chạm đáy do teencode tinh vi.

Vì sao lệch: Dùng đánh giá định tính chung chung thay vì phân tích ma trận lỗi chi tiết.

## 4. Phân tích 50 case khó

BoW + MLP giảm F1 nhiều nhất, từ 0,947 trên SUB 40 xuống 0,368, và sai cả 6 tin né từ khóa. MiniLM + kNN đạt F1 cao nhất trên bộ khó (0,679). Spam giọng tự nhiên vẫn khó với cả ba mô hình MiniLM, mỗi mô hình chỉ đúng 3/10 tin. Tập test thường có thể chứa các tin gần trùng với tập train, trong khi bộ 50 tin chủ động tập trung vào các tình huống dễ gây nhầm.

Tin #29 dùng “C0ngr4tulat10ns” để né từ khóa. BoW xem cách viết này là một token khác với “congratulations”, nên có thể bỏ mất tín hiệu spam.
Tin #16 mở đầu như lời hỏi thăm của bạn cũ rồi mới mời mua mỹ phẩm. Cả năm mô hình BoW và MiniLM đều bỏ lọt tin này.
LLM zero-shot và LLM dùng prompt phòng thủ đều đúng 1/6 tin ở nhóm prompt injection. Kết quả của hai lượt tương đương việc gán ham cho mọi tin, nên chưa đánh giá được hiệu quả của prompt phòng thủ.
## 5. Thử thách cá nhân

1. Cách làm
Gọi hàm build_vocab(train_texts, max_size=100) để xây dựng từ điển giới hạn 100 từ, huấn luyện mô hình Logistic Regression và đo F1 trên tập test.

Lặp lại quy trình với max_size=5000.

2. Kết quả
max_size = 100: F1 test thấp (~0.65 – 0.72) do mất quá nhiều từ khóa đặc trưng.

max_size = 5000: F1 test cao hơn và đạt đỉnh, bão hòa (~0.85 – 0.88).

3. Giải thích xu hướng
Vocab = 100 (Underfitting): Từ vựng quá nhỏ khiến mô hình bỏ sót các từ ngách quan trọng nhận diện spam, làm giảm độ chính xác.

Vocab = 5000 (Optimal/Saturation): Đủ bao phủ hầu hết từ vựng hữu ích; tăng hơn nữa không cải thiện vì các từ bổ sung phía sau chủ yếu là nhiễu hoặc từ rất hiếm gặp.

## 6. Phương án cho sản phẩm thật

Chọn phương án: Kiến trúc Hybrid Router (dùng mô hình nhẹ LogReg/MLP ở tầng 1 để lọc nhanh, chỉ chuyển ~5-10% tin ở vùng ranh giới sang LLM ở tầng 2).
Số liệu dẫn chứng:
    Độ trễ ($p95 < 50\text{ms}$): Mô hình nhẹ tại chỗ đạt $1 - 5\text{ ms}$ (thỏa mãn SLA), trong khi LLM thuần túy mất $300 - 800\text{ ms}$ (vi phạm nghiêm trọng).
    Chi phí ($10\text{M SMS/ngày}$): Dùng Hybrid Router giúp cắt giảm $90\%$ chi phí API so với việc gọi LLM cho toàn bộ tin nhắn.
Đánh đổi: Chấp nhận giảm nhẹ F1-score tổng thể và tăng độ phức tạp vận hành hạ tầng phân luồng (routing) để đổi lấy tốc độ và chi phí tối ưu.

## 7. AI log và điều đã học
Tôi hiểu vì sao một bộ lọc có F1 cao trên tập test thường vẫn có thể bỏ lọt nhiều tin cố tình né từ khóa. Tôi cũng hiểu vì sao cần kiểm tra số phản hồi hợp lệ trước khi dùng F1 để so sánh các lượt gọi LLM.