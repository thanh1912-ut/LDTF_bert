# Báo cáo validation 6 model - seed 42

- Trạng thái protocol: **ĐẠT**
- Model hoàn thành: **6/6**
- Model đứng đầu theo Macro F1: **B2_bert_finetuned_cls** (0.948616)
- Seed dùng chung: **42**
- Tập test: **chưa sử dụng**

## Xếp hạng validation

| rank | member_id | run | seed | best_epoch | val_f1_macro | val_accuracy | val_loss | train_hours | peak_vram_gb | protocol_ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000 | 1.000000 | B2_bert_finetuned_cls | 42.000000 | 3.000000 | 0.948616 | 0.948626 | 0.188012 | 2.157642 | 3.993800 | True |
| 2.000000 | 3.000000 | A11 | 42.000000 | 3.000000 | 0.947809 | 0.947874 | 0.185183 | 2.463422 | 4.182000 | True |
| 3.000000 | 1.000000 | A0 | 42.000000 | 3.000000 | 0.947372 | 0.947373 | 0.188514 | 2.277578 | 4.186300 | True |
| 4.000000 | 2.000000 | A3 | 42.000000 | 3.000000 | 0.946808 | 0.946872 | 0.188776 | 2.344108 | 4.180400 | True |
| 5.000000 | 2.000000 | A1 | 42.000000 | 3.000000 | 0.945998 | 0.946036 | 0.190123 | 2.334453 | 4.180400 | True |
| 6.000000 | 3.000000 | A4 | 42.000000 | 2.000000 | 0.945641 | 0.945702 | 0.175381 | 2.348097 | 4.180500 | True |

## Chênh lệch so với B2

| run | val_f1_macro | delta_f1_vs_B2 |
| --- | --- | --- |
| A11 | 0.947809 | -0.000807 |
| A0 | 0.947372 | -0.001244 |
| A3 | 0.946808 | -0.001808 |
| A1 | 0.945998 | -0.002618 |
| A4 | 0.945641 | -0.002975 |

## Cách đọc

Macro F1 cao hơn thể hiện model phân loại cân bằng hơn giữa bốn nhãn. Delta dương nghĩa là model tốt hơn B2 trên cùng seed 42. Vì mỗi model chỉ chạy một seed, báo cáo này chưa đo được độ ổn định khi đổi seed.

## Biểu đồ

- `01_validation_f1.png`: Macro F1 của 6 model.
- `02_f1_accuracy.png`: so sánh Macro F1 và accuracy.
- `03_delta_vs_B2.png`: mức tăng hoặc giảm so với B2.
- `04_runtime_hours.png`: thời gian train từng model.
