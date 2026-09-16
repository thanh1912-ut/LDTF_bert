"""Build the final-test notebook and a separate, checksummed source archive."""
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_colab_ready_folder import write_source_archive


def build():
    package = ROOT / "LDTF_4_MEMBERS"
    manifest = write_source_archive(package / "final_test/source/LDTF_final_test_source.zip")
    (package / "final_test/source/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config_path = package / "final_test/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(status="ready", evaluation_enabled=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cells = []
    def markdown(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=[text], id=f"cell-{len(cells)}"))
    def code(text):
        cells.append(dict(cell_type="code", metadata={}, source=textwrap.dedent(text).strip().splitlines(True),
                          execution_count=None, outputs=[], id=f"cell-{len(cells)}"))
    markdown("# Đánh giá cuối cùng B2 cho Member 4\nUpload `final_test/` vào `MyDrive/LDTF_4_MEMBERS/`, chọn GPU T4 và Run all. "
             "B2 đã được chọn bằng validation. Notebook sẽ truy cập tập test khi chạy cell 5. "
             "Chạy lại dùng dự đoán hoàn tất đã lưu; nếu ngắt giữa suy luận, chạy lại từ đầu lượt suy luận và ghi thêm ledger.")
    markdown("## 1. Kết nối Drive và cài môi trường")
    code('''
        import os, sys, json, hashlib, zipfile, subprocess, threading, time
        from pathlib import Path
        from tqdm.auto import tqdm
        RUN_TAG = "B2_colab_detailed_01"
        BOOTSTRAP_SAMPLES = 1000
        if not RUN_TAG or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in RUN_TAG):
            raise ValueError("RUN_TAG chỉ gồm chữ, số, gạch dưới hoặc gạch ngang.")
        # Local simulation uses an isolated fixture folder; normal Colab mounts Drive.
        simulation = os.environ.get("LDTF_COLAB_SIMULATION_ROOT")
        with tqdm(total=4, desc="Chuẩn bị môi trường", unit="bước") as progress:
            progress.set_postfix_str("Kết nối Drive")
            if simulation:
                MEMBER_ROOT = Path(simulation)
            else:
                from google.colab import drive
                drive.mount("/content/drive")
                MEMBER_ROOT = Path("/content/drive/MyDrive/LDTF_4_MEMBERS")
            progress.update(1)
            progress.set_postfix_str("Kiểm tra source")
            source = MEMBER_ROOT / "final_test/source/LDTF_final_test_source.zip"
            manifest = json.loads((source.parent / "manifest.json").read_text())
            if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["sha256"]:
                raise ValueError("Source ZIP sai checksum; upload lại final_test/source.")
            progress.update(1)
            progress.set_postfix_str("Giải nén source")
            import tempfile
            extraction = Path(tempfile.mkdtemp(prefix="ldtf_final_test_"))
            with zipfile.ZipFile(source) as archive:
                for item in tqdm(archive.infolist(), desc="Giải nén", unit="file", leave=False):
                    target = (extraction / item.filename).resolve()
                    if not target.is_relative_to(extraction.resolve()):
                        raise ValueError("Đường dẫn ZIP không hợp lệ.")
                    archive.extract(item, extraction)
            PROJECT = extraction / "LDTF_bert_source"
            progress.update(1)
            progress.set_postfix_str("Cài thư viện; xem log bên dưới")
            if not simulation:
                done = threading.Event()
                def heartbeat():
                    started = time.monotonic()
                    while not done.wait(1):
                        progress.set_postfix_str(f"Đang cài thư viện | {time.monotonic()-started:.0f}s")
                worker = threading.Thread(target=heartbeat, daemon=True)
                worker.start()
                process = None
                try:
                    process = subprocess.Popen([sys.executable, "-u", "-m", "pip", "install", "-r", str(PROJECT / "requirements.txt")],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                    if process.wait():
                        raise RuntimeError("Cài thư viện thất bại; xem log pip.")
                finally:
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.wait()
                    done.set(); worker.join()
            if str(PROJECT) not in sys.path:
                sys.path.insert(0, str(PROJECT))
            progress.update(1)
            progress.set_postfix_str("Hoàn tất")
        from experiments.final_test_package import FinalTest
        import torch
        if not simulation and not torch.cuda.is_available():
            raise RuntimeError("Chọn Runtime > Change runtime type > GPU T4 rồi chạy lại.")
        workflow = FinalTest(MEMBER_ROOT, overrides={
            "output_dir": f"final_test/colab_runs/{RUN_TAG}",
            "bundle_file": f"final_test/colab_runs/{RUN_TAG}_final_test.zip",
        })
        print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU mô phỏng")
        print("Đầu ra:", workflow.out)
    ''')
    for title, body in [
        ("2. Kiểm tra dữ liệu và checksum", "workflow.validate()"),
        ("3. Xác minh checkpoint B2", "workflow.check_checkpoint()"),
        ("4. Chuẩn bị tokenizer và model", "workflow.prepare()"),
        ("5. Chạy đánh giá chính thức trên test", "from experiments.final_eval import evaluate_package\nevaluate_package(workflow)"),
        ("6. Phân tích kết quả và vẽ biểu đồ", "metrics = workflow.analyze()\ntry:\n    from IPython.display import display, Image\nexcept ImportError:\n    print('Biểu đồ đã lưu tại:', workflow.out / 'figures')\nelse:\n    display(Image(filename=str(workflow.out / 'figures/confusion_matrix.png')))\n    display(Image(filename=str(workflow.out / 'figures/per_class_f1.png')))"),
        ("7. Phân tích chuyên sâu và khoảng tin cậy", '''
            from experiments.local_test_report import extended_report
            stats = extended_report(workflow, bootstrap_samples=BOOTSTRAP_SAMPLES)
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            try:
                from IPython.display import display, Image
            except ImportError:
                print("Biểu đồ:", workflow.out / "figures")
            else:
                for name in ("confusion_matrix_normalized.png", "calibration.png"):
                    display(Image(filename=str(workflow.out / "figures" / name)))
        '''),
        ("8. Xem chỉ số từng nhãn và các mẫu sai", '''
            import pandas as pd
            with tqdm(total=4, desc="Đọc bảng phân tích", unit="bảng") as progress:
                for name in ("metrics/per_class_metrics.csv", "metrics/confusion_pairs.csv",
                             "metrics/by_word_length.csv", "predictions/high_confidence_errors.csv"):
                    progress.set_postfix_str(name)
                    table = pd.read_csv(workflow.out / name)
                    print("\\n" + name)
                    try:
                        from IPython.display import display
                    except ImportError:
                        print(table.head(20).to_string(index=False))
                    else:
                        display(table.head(20))
                    progress.update(1)
                progress.set_postfix_str("Hoàn tất")
        '''),
        ("9. Xuất toàn bộ báo cáo và ZIP bàn giao", "bundle = workflow.export()\nprint('Báo cáo chi tiết:', workflow.out / 'reports/DETAILED_REPORT.html')\nprint('ZIP đã lưu trên Drive:', bundle)"),
    ]:
        markdown("## " + title); code(body)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}}, cells=cells)
    for path in (package / "colab_final_test_B2.ipynb", ROOT / "notebooks/colab_final_test_B2.ipynb"):
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("Built notebook and final-test source:", manifest)


if __name__ == "__main__":
    build()
