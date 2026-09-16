"""Generate the standalone local evaluation notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
def section(title, description, code):
    cells.append(dict(cell_type='markdown', id=f'm{len(cells)}', metadata={}, source=[f'## {title}\n\n{description}']))
    cells.append(dict(cell_type='code', id=f'c{len(cells)}', metadata={}, execution_count=None, outputs=[], source=code.splitlines(True)))

cells.append(dict(cell_type='markdown', id='intro', metadata={}, source=[
    '# Đánh giá B2 chuyên sâu trên GPU laptop\n\nChạy `scripts/setup_vscode_gpu.ps1` một lần trong PowerShell tại repo. '
    'Trong VS Code chọn kernel **LDTF GPU**, sau đó Run All. Không cần Google Drive. '
    'Mỗi RUN_TAG lưu riêng kết quả; đây là đánh giá lại B2 đã chốt, không dùng test để chọn cấu hình.']))
section('1. Thiết lập và nhận diện repo', 'Giữ MODE="evaluate" để suy luận lại trên GPU. MODE="report_only" chỉ phân tích dự đoán Colab đã có. Giữ RUN_TAG để tiếp tục báo cáo; đổi tên để tạo lượt đánh giá mới.', '''from pathlib import Path
import sys
MODE = "evaluate"
RUN_TAG = "B2_laptop_01"
BATCH_SIZE = 8
BOOTSTRAP_SAMPLES = 1000
candidates = list(Path.cwd().resolve().parents)
candidates.insert(0, Path.cwd().resolve())
candidates += [p / "LDTF_bert" for p in list(candidates)]
ROOT = next((p for p in candidates if (p / "experiments/local_test_report.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Mở notebook bên trong repo LDTF_bert.")
sys.path.insert(0, str(ROOT))
from experiments.local_test_report import LocalTest, extended_report, export_local
from experiments.final_test_package import activity
import torch
with activity("Kiểm tra kernel và GPU"):
    if MODE not in ("evaluate", "report_only"):
        raise ValueError("MODE phải là evaluate hoặc report_only")
    print("Python:", sys.executable, "\\nPyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Không có CUDA trong kernel")
    if MODE == "evaluate" and not torch.cuda.is_available():
        raise RuntimeError("Chạy scripts/setup_vscode_gpu.ps1 rồi chọn kernel LDTF GPU; kernel hiện tại là CPU.")
    workflow = LocalTest(ROOT / "LDTF_4_MEMBERS", RUN_TAG, BATCH_SIZE)
    print("Đầu ra:", workflow.out)
''')
section('2. Kiểm tra đầu vào', 'Kiểm tra checkpoint, dữ liệu và bằng chứng chọn B2 bằng validation.', 'workflow.validate()\nworkflow.check_checkpoint()\n')
section('3. Chuẩn bị model', 'Nạp trọng số lên GPU; lần đầu có thể cần mạng để tải tokenizer và backbone BERT.', '''if MODE == "evaluate":
    workflow.prepare()
else:
    workflow.import_previous()
''')
section('4. Dự đoán test', 'Theo dõi batch và số mẫu. Nếu thiếu VRAM, restart kernel, đặt BATCH_SIZE=4 và đổi RUN_TAG. Lượt truy cập test được ghi log.', '''from experiments.final_eval import evaluate_package
if MODE == "evaluate":
    evaluate_package(workflow)
else:
    with activity("Dùng dự đoán đã xác minh"):
        print("Chế độ phân tích; không suy luận lại.")
''')
section('5. Chỉ số cơ bản', 'Accuracy, Macro F1, loss, precision/recall/F1 từng nhãn và confusion matrix.', '''metrics = workflow.analyze()
import pandas as pd
from IPython.display import display, Image, HTML
display(pd.read_csv(workflow.out / "metrics/per_class_metrics.csv"))
display(Image(filename=str(workflow.out / "figures/confusion_matrix.png")))
''')
section('6. Phân tích chuyên sâu', 'Khoảng tin cậy bootstrap, hiệu chuẩn xác suất, ma trận chuẩn hóa, độ dài và lỗi tự tin cao. Bootstrap chỉ dùng dự đoán đã lưu.', '''extended = extended_report(workflow, BOOTSTRAP_SAMPLES)
display(pd.DataFrame([extended]))
for name in ["confusion_matrix_normalized.png", "calibration.png", "per_class_f1.png"]:
    display(Image(filename=str(workflow.out / "figures" / name)))
''')
section('7. Kiểm tra mẫu sai', 'Đọc các tin bị nhầm để mô tả hạn chế. Không sửa mô hình theo tập test.', '''with activity("Đọc bảng lỗi"):
    errors = pd.read_csv(workflow.out / "predictions/high_confidence_errors.csv")
    display(errors.head(20))
    display(pd.read_csv(workflow.out / "metrics/confusion_pairs.csv").head(6))
''')
section('8. Xuất báo cáo', 'HTML có thể mở bằng trình duyệt; ZIP chứa toàn bộ kết quả và checksum.', '''bundle = export_local(workflow)
print("Báo cáo HTML:", workflow.out / "reports/DETAILED_REPORT.html")
print("Báo cáo Markdown:", workflow.out / "reports/DETAILED_REPORT.md")
print("ZIP:", bundle)
''')
notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name':'LDTF GPU','language':'python','name':'ldtf-gpu'}},cells=cells)
target = ROOT / 'notebooks/vscode_final_test_B2_detailed.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(target)
