# Danh sách kiểm tra nhanh

Nguồn kết quả chính là `reports/multisource/final_data_report.json`.

- [x] Ba nguồn được đọc thành công.
- [x] Nhãn chỉ thuộc 0 đến 3.
- [x] Văn bản cuối không rỗng.
- [x] Không còn artifact HTML và backslash đã cấu hình.
- [x] Train cuối không còn hash trùng hoàn toàn.
- [x] Dữ liệu bổ sung không trùng validation/test.
- [x] Phần bổ sung cân bằng 324 mẫu mỗi nhãn.
- [x] Validation và test giữ nguyên fingerprint.
- [x] Tokenizer khớp `bert-base-uncased`.
- [x] Batch đầu vào hợp lệ.
- [ ] Hoàn tất duyệt 100 mẫu trong `manual_review_samples.csv` trước khi công bố.

Giải thích và số liệu chi tiết nằm tại `../../docs/data/DATA_QUALITY.md`.
