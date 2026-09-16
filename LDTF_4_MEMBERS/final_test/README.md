# Đánh giá cuối cùng trên tập test

Trạng thái: đã triển khai chín cell, sẵn sàng chạy đánh giá chi tiết trên Colab.

Đọc `KE_HOACH.md` trước, sau đó xem `config.json` và notebook
`../colab_final_test_B2.ipynb`. Chọn GPU T4 rồi Run all; cell 5 sẽ đánh giá test.

```text
LDTF_4_MEMBERS/
├── colab_final_test_B2.ipynb
└── final_test/
    ├── README.md
    ├── KE_HOACH.md
    ├── config.json
    ├── source/
    │   ├── LDTF_final_test_source.zip
    │   └── manifest.json
    ├── inputs/
    │   ├── B2_bert_finetuned_cls_seed42/best.pt
    │   ├── research_test.parquet
    │   └── selection/
    │       ├── validation_ranking.csv
    │       ├── protocol_violations.json
    │       ├── run_summary.json
    │       └── team_job_metadata.json
    └── B2_bert_finetuned_cls_seed42/
        ├── metrics/
        ├── predictions/
        ├── figures/
        ├── logs/
        └── reports/
```

Notebook mới lưu vào `colab_runs/B2_colab_detailed_01/`, không ghi đè kết quả cũ.
Checkpoint được sao chép từ file người dùng cung cấp; file gốc vẫn được giữ.
`best.pt` đã được kiểm tra metadata: B2, seed 42, epoch 3, Macro F1 validation
0.9486156611425511. SHA-256 trong config dùng để đối chiếu khi upload lên Drive.

Member 4 đặt cả `final_test/` và notebook cạnh
`source/` trong `MyDrive/LDTF_4_MEMBERS/`. Chỉ cần checkpoint B2 để thực hiện
protocol đã chốt. Không dùng thư mục `incoming/download_staging` làm đầu vào.
Notebook dùng source ZIP riêng trong `final_test/source/`, kiểm tra SHA-256 trước
khi giải nén. Không cần thay source ZIP dùng để train hoặc phân tích validation.

## Cách chạy

1. Upload nguyên `final_test/` vào `MyDrive/LDTF_4_MEMBERS/`.
2. Kiểm tra `inputs/B2_bert_finetuned_cls_seed42/best.pt` đã upload xong.
3. Mở notebook mới `colab_final_test_B2.ipynb`, chọn GPU T4 rồi Run all.
4. Cấp quyền Drive ở cell 1. Các bước cài đặt/nạp model có trạng thái thời gian;
   kiểm tra checksum có tiến trình byte; suy luận có tiến trình batch.
5. Cell 6 tính chỉ số cơ bản; cell 7 tính khoảng tin cậy bootstrap, ECE/Brier
   và vẽ biểu đồ; cell 8 hiển thị bảng từng nhãn và mẫu sai; cell 9 đóng gói.
6. Xem `colab_runs/B2_colab_detailed_01/reports/DETAILED_REPORT.html`
   và ZIP `colab_runs/B2_colab_detailed_01_final_test.zip` trong `final_test/`.
   Giải nén cả ZIP trước khi mở HTML để các biểu đồ hiển thị đúng.

Nếu đã upload đầu vào trước đây, chỉ cần thay notebook và **cả hai file trong
`final_test/source/`** bằng bản mới. Giữ nguyên `config.json` và `inputs/`.
Không chỉ upload notebook vì phần xử lý được nạp từ source ZIP.
Cell 1 có `RUN_TAG` để đặt tên lượt đánh giá và `BOOTSTRAP_SAMPLES=1000`.
Giữ nguyên RUN_TAG khi tiếp tục lượt bị ngắt; đổi tên chỉ khi muốn đánh giá lại
vào thư mục mới. Bootstrap không train lại và không đại diện cho nhiều seed.

Nếu chạy lại khi đã có dự đoán hợp lệ, notebook tái tạo báo cáo từ dự đoán cũ.
Nếu ngắt giữa suy luận, chạy lại sẽ suy luận từ đầu và thêm lần truy cập vào ledger;
đây không phải resume theo batch. Nếu sai checksum hoặc cấu hình thay đổi, notebook
dừng để tránh ghép nhầm kết quả. Không chỉnh config nhằm tối ưu theo điểm test.
Khi cập nhật source trong cùng phiên Colab, restart runtime trước khi Run all.

Không cần train lại. Inference dùng float32 để ổn định và tương thích T4.
Log bước chính lưu tại `logs/console.log`, lỗi cell vẫn hiển thị trực tiếp trong Colab.

Checkpoint `.pt` đã được Git ignore. Dữ liệu test sao chép và các đầu ra lớn
trong bộ bàn giao này cũng được ignore cục bộ bằng `final_test/.gitignore`.
