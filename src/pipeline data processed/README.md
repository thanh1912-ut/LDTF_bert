# Pipeline dữ liệu đa nguồn

Pipeline mở rộng tập train AG News cố định bằng dữ liệu BBC News và HuffPost đã
lọc, đồng thời giữ nguyên validation và test để các thí nghiệm có thể so sánh
công bằng.

## Nguồn nhãn bổ sung

| Nhãn chung | Nguồn | Chuyên mục được nhận |
| --- | --- | --- |
| World | HuffPost | `WORLD NEWS` |
| Sports | BBC News | `sport` |
| Business | BBC News | `business` |
| Sci/Tech | BBC News | `tech` |

Quy tắc nguồn, giới hạn số mẫu, ngưỡng gần trùng và tokenizer nằm trong
`configs/pipeline_config.json`.

## Chạy pipeline

Từ thư mục dự án `LDTF_bert` trên Windows:

```powershell
& '.venv/Scripts/python.exe' 'src/pipeline data processed/scripts/run_pipeline.py'
```

Ba file Parquet sẵn sàng cho model được tạo tại `data/processed` bên trong
folder pipeline này. Báo cáo nằm tại `reports/multisource`, còn checksum, danh
sách ID và lịch sử loại hàng nằm tại `manifests/multisource`.

## Dùng dữ liệu mới để huấn luyện

Trước khi chạy model, trỏ project đến thư mục data của pipeline:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path -LiteralPath 'src/pipeline data processed/data').Path
```

Project sẽ đọc:

- `research_train.parquet`;
- `research_validation.parquet`;
- `research_test.parquet`.

## Trạng thái hiện tại

- Train: 109.031 mẫu, gồm 1.296 mẫu bổ sung cân bằng.
- Validation: 11.971 mẫu AG News được giữ nguyên.
- Test: 7.600 mẫu AG News được giữ nguyên.
- Kiểm tra kỹ thuật tự động: `PASS`.
- Kiểm thử pipeline đa nguồn: 5/5 đạt.

Trước khi công bố một phiên bản dataset, cần đọc
`reports/multisource/manual_review_samples.csv`. Báo cáo tự động đầy đủ nằm tại
`reports/multisource/final_data_report.json`.

## Tài liệu chi tiết

- `../../docs/data/README.md`: index và thứ tự đọc đề xuất;
- `../../docs/data/DATASET_CARD.md`: thành phần, mục đích và giới hạn dataset;
- `../../docs/data/DATA_DICTIONARY.md`: ý nghĩa từng cột;
- `../../docs/data/PROCESSING_PIPELINE.md`: thuật toán và thứ tự xử lý;
- `../../docs/data/DATA_QUALITY.md`: kết quả kiểm tra chất lượng;
- `../../docs/data/REPRODUCIBILITY.md`: checksum và cách tái lập.
