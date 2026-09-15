# Hướng dẫn đọc tài liệu dữ liệu

Đây là trang bắt đầu cho toàn bộ tài liệu về bộ dữ liệu `multisource-v1`.

## Thứ tự đọc đề xuất

### 1. Hiểu bộ dữ liệu đang có gì

Đọc [DATASET_CARD.md](DATASET_CARD.md) trước.

File này trả lời các câu hỏi:

- dữ liệu đến từ đâu;
- có bao nhiêu mẫu train, validation và test;
- BBC và HuffPost được ánh xạ nhãn như thế nào;
- dataset dùng được cho việc gì;
- dataset còn hạn chế gì.

Đây là tài liệu quan trọng nhất cho người mới và phần mô tả dữ liệu trong báo
cáo hoặc khóa luận.

### 2. Hiểu ý nghĩa từng cột

Đọc [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

File này giải thích các cột như `text`, `label`, `source_dataset`,
`title_clean` và `text_hash`. Đọc file này khi muốn mở Parquet, xem một hàng dữ
liệu hoặc viết code sử dụng dataset.

### 3. Hiểu dữ liệu được xử lý như thế nào

Đọc [PROCESSING_PIPELINE.md](PROCESSING_PIPELINE.md).

File này mô tả thứ tự:

```text
đọc dữ liệu → chuẩn hóa → làm sạch → loại trùng
→ chống rò rỉ → lấy mẫu cân bằng → xuất Parquet
```

Đọc file này khi cần giải thích phương pháp tiền xử lý hoặc chỉnh sửa pipeline.

### 4. Kiểm tra dữ liệu có đạt chất lượng không

Đọc [DATA_QUALITY.md](DATA_QUALITY.md).

File này giải thích trạng thái `PASS`, số hàng trùng bị loại, phân bố nhãn, độ
dài token, fingerprint của tập đánh giá và các rủi ro còn lại.

Đọc file này trước khi bắt đầu huấn luyện hoặc trình bày kết quả thực nghiệm.

### 5. Chạy lại đúng phiên bản dữ liệu

Đọc [REPRODUCIBILITY.md](REPRODUCIBILITY.md) sau cùng.

File này chứa seed, checksum, cấu hình, lệnh chạy và cách trỏ model đến đầu ra
pipeline. Nó dành cho lúc cần tái lập dataset trên máy khác hoặc chứng minh hai
lần chạy sử dụng cùng dữ liệu.

## Lộ trình ngắn cho người mới

Nếu chỉ muốn hiểu tổng quan, đọc theo thứ tự:

```text
DATASET_CARD.md → DATA_QUALITY.md
```

Nếu muốn hiểu để viết báo cáo:

```text
DATASET_CARD.md → PROCESSING_PIPELINE.md
→ DATA_QUALITY.md → DATA_DICTIONARY.md
```

Nếu muốn chạy hoặc sửa code:

```text
DATA_DICTIONARY.md → PROCESSING_PIPELINE.md
→ REPRODUCIBILITY.md → DATA_QUALITY.md
```

## Bản đồ tài liệu

| Tài liệu | Câu hỏi chính | Mức độ |
| --- | --- | --- |
| [DATASET_CARD.md](DATASET_CARD.md) | Dataset gồm những gì? | Cơ bản |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Mỗi cột có nghĩa gì? | Cơ bản |
| [PROCESSING_PIPELINE.md](PROCESSING_PIPELINE.md) | Dữ liệu được xử lý ra sao? | Trung bình |
| [DATA_QUALITY.md](DATA_QUALITY.md) | Dữ liệu đã đạt yêu cầu chưa? | Trung bình |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Làm sao tạo lại đúng dataset? | Kỹ thuật |

## Tài liệu vận hành liên quan

- `../../HUONG_DAN_TRAIN_DATA_MOI.md`: hướng dẫn dùng dataset mới để train,
  tiếp tục checkpoint và đánh giá cuối cùng;
- `../../notebooks/colab_4_members_multisource.ipynb`: notebook dùng chung cho
  ba thành viên train và một thành viên phân tích;
- `../../configs/colab_4_members.json`: seed, model, checksum và protocol chung
  của nhóm Colab;
- `src/pipeline data processed/README.md`: lệnh chạy nhanh và vị trí đầu ra;
- `src/pipeline data processed/kiểm tra.md`: checklist ngắn;
- `src/pipeline data processed/configs/pipeline_config.json`: cấu hình thực tế;
- `src/pipeline data processed/reports/multisource/final_data_report.json`:
  kết quả kiểm tra tự động gần nhất.
