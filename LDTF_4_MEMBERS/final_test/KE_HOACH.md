# Kế hoạch hoàn thành final-test cho Member 4

Đã triển khai notebook bảy cell và source ZIP riêng. Các mục dưới mô tả thiết kế;
hướng dẫn chạy thực tế ở `README.md`. Chưa chạy suy luận trên tập test thật tại máy.

## 1. Chốt lựa chọn trước khi xem test

- Đã chọn B2 bằng Macro F1 validation trong sáu model, cùng seed 42.
- Đã kiểm tra checkpoint có kiến trúc BertPooledClassifier, pooling cls,
  bert-base-uncased, bốn lớp, epoch 3 và seed 42.
- Đã lưu cấu hình đường dẫn và checksum; chưa mở khóa loader test.
- Giữ bản sao bảng xếp hạng và metadata làm bằng chứng lựa chọn.

## 2. Triển khai notebook bảy cell

1. Kết nối Drive, xác định thư mục gốc, chuẩn bị source và thư viện.
2. Đọc config, kiểm tra đủ đầu vào và SHA-256 với thanh tiến trình.
3. Đối chiếu checkpoint với summary, seed, epoch, kiến trúc, chữ ký dữ liệu
   và quyết định chọn B2. Kiểm tra kết quả cũ để tránh chạy lại vô tình.
4. Chuẩn bị tokenizer, model và GPU; kiểm tra nạp trọng số strict.
5. Chạy final evaluation qua `experiments/final_eval.py` và
   `src.guard.official_test_loader`; ghi lý do truy cập, checksum và log về Drive.
6. Dùng dự đoán đã lưu để tính chỉ số theo nhãn, vẽ confusion matrix,
   phân tích mẫu sai và đối chiếu validation với test.
7. Xuất báo cáo tiếng Việt, manifest đầu ra và ZIP bàn giao.

Mọi bước cần hiện trạng thái; bước dự đoán cần hiển thị tiến trình theo batch.
Chỉ mở khóa test tại bước đánh giá sau khi kiểm tra đầu vào thành công.

## 3. Tận dụng mã hiện có

- `src/evaluate.py`: dựng lại model từ architecture và nạp checkpoint.
- `experiments/final_eval.py`: điểm vào đánh giá test chính thức.
- `src/guard.py`: kiểm soát và ghi lịch sử truy cập tập test.
- Cần nối đường dẫn input/output mới với cấu hình hiện có, lưu ledger bền vững
  trên Drive và bổ sung xuất báo cáo, biểu đồ. Không xây loader test thứ hai.
- Lưu dự đoán dạng NPZ theo mã hiện tại; xuất bảng mẫu sai từ cùng lượt dự đoán.

## 4. Đầu ra dự kiến

Trong `final_test/B2_bert_finetuned_cls_seed42/`:

- `metrics/test_metrics.json`: Accuracy, Macro F1, loss và metadata checkpoint.
- `metrics/per_class_metrics.csv`: precision, recall, F1, support của bốn nhãn.
- `predictions/test_predictions.npz`: nhãn thật và dự đoán, cần quy định rõ schema.
- `predictions/errors.csv`: các mẫu sai có chỉ số dòng để truy vết.
- `figures/confusion_matrix.png`: ma trận nhầm lẫn.
- `logs/official_test_access.jsonl`: nhật ký truy cập test.
- `logs/console.log`: log thực thi.
- `reports/FINAL_TEST_REPORT.md`: báo cáo tiếng Việt và giới hạn kết luận.
- `reports/artifact_manifest.json`: danh sách file và checksum.

## 5. Kiểm thử trước khi mở test

- Kiểm tra notebook hợp lệ, cú pháp các cell và đường dẫn Drive/local.
- Dùng dữ liệu giả để thử tính metrics, xuất biểu đồ và xử lý lỗi.
- Kiểm tra từ chối checkpoint khác model/seed, checksum sai, thiếu đầu vào.
- Kiểm tra loader vẫn khóa trước bước final evaluation.
- Kiểm tra ghi file nguyên tử; chỉ đánh dấu hoàn thành khi đủ đầu ra.
- Chạy lại notebook khi đã hoàn tất phải đọc kết quả cũ; nếu gián đoạn, phân biệt
  trạng thái chưa hoàn thành và hoàn thành, ghi nhận lần thử lại trong ledger.

## 6. Thực hiện và diễn giải

Sau khi triển khai, cập nhật bộ source bàn giao, chạy final-test trên Colab,
đối chiếu số mẫu với cấu hình (dự kiến 7.600), rồi kiểm tra toàn bộ artifact.
Không điều chỉnh hyperparameter hoặc chọn lại model dựa trên điểm test.
Một seed chưa đủ kết luận độ ổn định hoặc ý nghĩa thống kê của chênh lệch nhỏ.
Test là tập AG News nên kết quả không đại diện đầy đủ cho mọi nguồn BBC/HuffPost.

Việc đánh giá nhiều model đã chốt trước trên test có thể hợp lệ cho nghiên cứu
so sánh nếu không dùng điểm test để chọn hoặc chỉnh model. Protocol của bộ bàn
giao này giới hạn B2; phát biểu trước đó rằng luôn cấm test nhiều model là quá tuyệt đối.
