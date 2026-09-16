# Đánh giá B2 trên GPU laptop

Notebook: `notebooks/vscode_final_test_B2_detailed.ipynb`.

Máy được kiểm tra có RTX 3050 4 GB. `.venv` hiện là PyTorch CPU, vì vậy cần
chuẩn bị kernel riêng trước lần chạy GPU đầu tiên. Trong terminal PowerShell,
đứng ở thư mục `LDTF_bert`, chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_vscode_gpu.ps1
```

Script tạo `.venv-gpu`, cài PyTorch CUDA, các thư viện của repo và kernel Jupyter.
Cần mạng và vài GB dung lượng tải. Chọn kernel **LDTF GPU** trong VS Code.
Lệnh PyTorch CUDA 12.6 tham khảo [hướng dẫn chính thức](https://pytorch.org/get-started/previous-versions/#v2-11-0).

Giữ `MODE="evaluate"`, `BATCH_SIZE=8` và bấm Run All. Input đã nằm trong
`LDTF_4_MEMBERS/final_test/inputs/`; không cần Google Drive. Nếu thiếu VRAM,
restart kernel và giảm batch còn 4, đổi RUN_TAG để tạo lượt mới.

Đầu ra mặc định:

```text
LDTF_4_MEMBERS/final_test/local_runs/B2_laptop_01/
  metrics/       # Chỉ số từng lớp, ECE, Brier, CI95, cặp nhãn nhầm
  predictions/   # Toàn bộ dự đoán và lỗi kèm xác suất
  figures/       # Confusion matrix, reliability, F1 từng nhãn
  logs/          # Môi trường, checkpoint, lịch sử truy cập test
  reports/DETAILED_REPORT.html
  reports/DETAILED_REPORT.md
```

ZIP nằm cạnh thư mục lượt chạy. Giữ RUN_TAG để dùng lại dự đoán đã hoàn tất;
đổi RUN_TAG nếu chủ động đánh giá lại. Nếu ngắt giữa suy luận, chạy lại từ đầu
lượt suy luận, không resume theo batch. Không thay model hoặc checkpoint theo
kết quả test. Việc đánh giá lại này không tạo ra một test set độc lập mới.

`MODE="report_only"` xác minh manifest và dùng dự đoán Colab đã lưu để tạo báo
cáo mà không cần GPU. Chế độ này không phải lần đo hiệu năng laptop.
Khoảng tin cậy bootstrap chỉ phản ánh lấy mẫu test với model cố định; không
thay thế việc huấn luyện nhiều seed. Xác suất softmax không phải bảo đảm đúng.
