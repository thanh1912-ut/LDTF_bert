"""Generate the Colab training notebook used by members 1, 2, and 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "notebooks" / "colab_4_members_multisource.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def build_cells() -> list[dict]:
    return [
        markdown("""# LDTF-BERT - Notebook train cho Member 1, 2, 3

Mỗi người upload nguyên folder `LDTF_4_MEMBERS` vào Google Drive cá nhân, chỉ sửa `MEMBER_ID`, bật **T4 GPU**, rồi chọn **Runtime > Run all**.

- Member 1: `B2_bert_finetuned_cls`, `A0` với seed 42
- Member 2: `A1`, `A3` với seed 42
- Member 3: `A4`, `A11` với seed 42

Kết quả được ghi trực tiếp lên Drive. Sau mỗi epoch hoàn tất, `last.pt`, `train_log.jsonl` và `val_metrics.json` được cập nhật. Khi Colab ngắt, Run all lại sẽ tiếp tục từ `last.pt`. Nếu ngắt giữa epoch, epoch đang dở sẽ chạy lại. `best.pt` luôn giữ mô hình có validation Macro F1 tốt nhất. Tập test không được dùng trong notebook này.
"""),
        markdown("## 1. Chỉ sửa MEMBER_ID ở cell này"),
        code("""from tqdm.auto import tqdm

with tqdm(total=1, desc="Cấu hình thành viên", unit="bước") as progress:
    progress.set_postfix_str("Trạng thái: đang kiểm tra MEMBER_ID")
    MEMBER_ID = 1  # Chỉ đổi thành 1, 2 hoặc 3
    MEMBER_DRIVE_ROOT = "/content/drive/MyDrive/LDTF_4_MEMBERS"
    SOURCE_ARCHIVE_NAME = "LDTF_bert_source.zip"
    EXTRACT_DIR = "/content/ldtf_source"
    assert MEMBER_ID in (1, 2, 3), "MEMBER_ID phải là 1, 2 hoặc 3"
    progress.update(1)
    progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
print("Đã chọn MEMBER_ID =", MEMBER_ID)
"""),
        markdown("## 2. Kết nối Drive và nạp source"),
        code("""from google.colab import drive
from tqdm.auto import tqdm

with tqdm(total=4, desc="Chuẩn bị source", unit="bước") as progress:
    progress.set_postfix_str("Trạng thái: đang kết nối Drive")
    drive.mount("/content/drive")
    progress.update(1)
    import hashlib, json, os, shutil, subprocess, sys, zipfile
    from datetime import datetime, timezone
    from pathlib import Path
    MEMBER_ROOT = Path(MEMBER_DRIVE_ROOT)
    for relative in ("source", "data/processed", "results", "exports"):
        (MEMBER_ROOT / relative).mkdir(parents=True, exist_ok=True)
    SOURCE_ARCHIVE = MEMBER_ROOT / "source" / SOURCE_ARCHIVE_NAME
    assert SOURCE_ARCHIVE.is_file(), f"Thiếu source ZIP: {SOURCE_ARCHIVE}"
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang kiểm tra source")

    def sha256_file(path, chunk_size=1024 * 1024):
        digest = hashlib.sha256()
        total = max(1, Path(path).stat().st_size)
        with Path(path).open("rb") as handle, tqdm(total=total, desc=f"SHA256 {Path(path).name}", unit="B", unit_scale=True, leave=False) as bar:
            bar.set_postfix_str("Trạng thái: đang tính checksum")
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
                bar.update(len(chunk))
        return digest.hexdigest()

    SOURCE_SHA256 = sha256_file(SOURCE_ARCHIVE)
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang giải nén source")
    EXTRACT_ROOT = Path(EXTRACT_DIR)
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    EXTRACT_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        root = EXTRACT_ROOT.resolve()
        members = archive.infolist()
        for item in members:
            target = (EXTRACT_ROOT / item.filename).resolve()
            assert target == root or root in target.parents, f"ZIP không an toàn: {item.filename}"
        for item in tqdm(members, desc="Giải nén source", unit="file", leave=False):
            archive.extract(item, EXTRACT_ROOT)
    candidates = [EXTRACT_ROOT] + [p for p in EXTRACT_ROOT.iterdir() if p.is_dir()]
    PROJECT = next((p for p in candidates if (p / "src").is_dir()), None)
    assert PROJECT is not None, "Source ZIP không chứa src/"
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    progress.update(1)
    progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
print("Project:", PROJECT)
print("Source SHA-256:", SOURCE_SHA256)
"""),
        markdown("## 3. Cài thư viện và đọc cấu hình chung"),
        code("""from tqdm.auto import tqdm

with tqdm(total=2, desc="Cài đặt môi trường", unit="bước") as progress:
    progress.set_postfix_str("Trạng thái: đang cài thư viện")
    print("Đang cài thư viện. Log pip được hiển thị ngay bên dưới.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(PROJECT / "requirements.txt")], check=True)
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang đọc config")
    CONFIG_PATH = PROJECT / "configs" / "colab_4_members.json"
    TEAM_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    CONFIG_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    MEMBER = TEAM_CONFIG["members"][str(MEMBER_ID)]
    assert MEMBER["role"] == "trainer"
    RESULTS_ROOT = MEMBER_ROOT / TEAM_CONFIG["storage"]["results_subdir"]
    EXPORT_ROOT = MEMBER_ROOT / TEAM_CONFIG["storage"]["exports_subdir"]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    progress.update(1)
    progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
print("Phân công:", MEMBER["label"])
print("Config SHA-256:", CONFIG_SHA256)
print("Kết quả lưu tại:", RESULTS_ROOT)
"""),
        markdown("## 4. Kiểm tra T4 và đưa dữ liệu vào ổ Colab"),
        code("""from tqdm.auto import tqdm
import pandas as pd
import torch

def copy_with_progress(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as src, temporary.open("wb") as dst, tqdm(total=source.stat().st_size, desc=f"Copy {source.name}", unit="B", unit_scale=True, leave=False) as bar:
        bar.set_postfix_str("Trạng thái: đang copy")
        for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
            dst.write(chunk)
            bar.update(len(chunk))
    temporary.replace(destination)

with tqdm(total=4, desc="Kiểm tra GPU và dữ liệu", unit="bước") as progress:
    progress.set_postfix_str("Trạng thái: đang kiểm tra GPU")
    assert torch.cuda.is_available(), "Chưa bật GPU: Runtime > Change runtime type > T4 GPU"
    GPU_NAME = torch.cuda.get_device_name(0)
    assert "T4" in GPU_NAME.upper(), f"Nhóm yêu cầu T4, hiện tại là {GPU_NAME}"
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang chuẩn bị đường dẫn")
    dataset_cfg = TEAM_CONFIG["dataset"]
    drive_data = MEMBER_ROOT / dataset_cfg["drive_processed_subdir"]
    local_root = Path(dataset_cfg["local_data_root"])
    local_processed = local_root / "processed"
    local_processed.mkdir(parents=True, exist_ok=True)
    required = (dataset_cfg["train_file"], dataset_cfg["validation_file"])
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang xác minh checksum")
    for filename in tqdm(required, desc="Xác minh và copy data", unit="file", leave=False):
        source = drive_data / filename
        destination = local_processed / filename
        assert source.is_file(), f"Thiếu dữ liệu: {source}"
        expected = dataset_cfg["sha256"][filename]
        assert sha256_file(source) == expected, f"Checksum sai: {source}"
        if not destination.exists() or sha256_file(destination) != expected:
            copy_with_progress(source, destination)
        assert sha256_file(destination) == expected
    progress.update(1)
    progress.set_postfix_str("Trạng thái: đang kiểm tra số dòng và nhãn")
    for split, filename in (("train", required[0]), ("validation", required[1])):
        labels = pd.read_parquet(local_processed / filename, columns=["label"])["label"]
        counts = labels.value_counts().sort_index().tolist()
        assert len(labels) == dataset_cfg["expected_rows"][split]
        assert counts == dataset_cfg["expected_label_counts"][split]
        print(f"{split}: {len(labels):,} dòng, nhãn {counts}")
    assert not (local_processed / dataset_cfg["sealed_test_file"]).exists()
    os.environ["LDTF_DATA_DIR"] = str(local_root)
    progress.update(1)
    progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
usage = shutil.disk_usage(MEMBER_ROOT)
print("GPU:", GPU_NAME)
print(f"Drive còn trống khoảng {usage.free / (1024**3):.2f} GB")
if usage.free < 4 * 1024**3:
    print("CẢNH BÁO: 2 cặp best.pt + last.pt có thể cần khoảng 3-4 GB.")
"""),
        markdown("## 5. Smoke test"),
        code("""from tqdm.auto import tqdm

with tqdm(total=1, desc="Smoke test", unit="bước") as progress:
    progress.set_postfix_str("Trạng thái: đang kiểm tra source")
    marker_dir = RESULTS_ROOT / "_team_checks"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"member_{MEMBER_ID}_{SOURCE_SHA256}.smoke_passed"
    if marker.exists():
        print("Smoke test đã PASS với source này, bỏ qua.")
    else:
        subprocess.run([sys.executable, "-m", "scripts.smoke_test", "--quick"], cwd=PROJECT, check=True)
        marker.write_text("PASS\\n", encoding="utf-8")
    progress.update(1)
    progress.set_postfix_str("Trạng thái: PASS", refresh=True)
print("Smoke test: PASS")
"""),
        markdown("""## 6. Train và tự động resume

Thanh ngoài theo dõi 2 model được phân công; chương trình train có thanh tiến trình theo batch và epoch. Job đã có `run_summary.json` hợp lệ sẽ được bỏ qua; job có `last.pt` sẽ tự thêm `--resume`.
"""),
        code("""from tqdm.auto import tqdm

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def run_with_live_output(command, output_dir, queue):
    import codecs
    import queue as queue_module
    import threading
    import time

    messages = queue_module.Queue()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=PROJECT, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    def read_output():
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                messages.put(chunk)
        except Exception as error:
            messages.put(error)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    log_path = output_dir / "console.log"
    print(f"PID: {process.pid} | Log trực tiếp: {log_path}", flush=True)
    try:
        with log_path.open("ab") as log:
            while True:
                try:
                    chunk = messages.get(timeout=2)
                except queue_module.Empty:
                    elapsed = int(time.monotonic() - started)
                    queue.set_postfix_str(
                        f"{output_dir.name} | {elapsed}s | đang chờ log tiếp theo"
                    )
                    queue.refresh()
                    continue
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                log.write(chunk)
                log.flush()
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            sys.stdout.write(decoder.decode(b"", final=True))
            sys.stdout.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=5)
        process.stdout.close()

def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

protocol = TEAM_CONFIG["protocol"]
assert protocol["delete_last_checkpoint_after_success"] is False
seed = int(MEMBER["seed"])
jobs = list(MEMBER["assigned_runs"])
assert len(jobs) == 2 and set(jobs) <= set(protocol["runs"])
with tqdm(total=len(jobs), desc=f"Member {MEMBER_ID} - seed {seed}", unit="model") as queue:
    for run_name in jobs:
        run_label = f"{run_name}_seed{seed}"
        queue.set_postfix_str(f"Trạng thái: đang kiểm tra {run_label}")
        output_dir = RESULTS_ROOT / run_label
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "run_summary.json"
        last_path = output_dir / "last.pt"
        metadata_path = output_dir / "team_job_metadata.json"
        previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            signature = summary.get("data_signature", {})
            assert summary.get("base_run") == run_name and summary.get("seed") == seed
            assert signature.get("train_sha256") == dataset_cfg["sha256"][dataset_cfg["train_file"]]
            assert signature.get("validation_sha256") == dataset_cfg["sha256"][dataset_cfg["validation_file"]]
            assert previous.get("source_archive_sha256") == SOURCE_SHA256
            assert previous.get("config_sha256") == CONFIG_SHA256
            previous.update(status="complete", run_summary_sha256=sha256_file(summary_path))
            write_json(metadata_path, previous)
            assert last_path.exists(), f"Thiếu last.pt cần giữ lại: {run_label}"
            queue.set_postfix_str(f"Trạng thái: đã có {run_label}")
            queue.update(1)
            continue
        resume = last_path.exists()
        if resume:
            assert previous, "Có last.pt nhưng thiếu team_job_metadata.json"
            assert previous.get("source_archive_sha256") == SOURCE_SHA256
            assert previous.get("config_sha256") == CONFIG_SHA256
            assert previous.get("member_id") == MEMBER_ID
            print(f"\\nTiếp tục {run_label} từ epoch hoàn tất gần nhất.")
            queue.set_postfix_str(f"Trạng thái: đang resume {run_label}")
        else:
            leftovers = [n for n in ("best.pt", "train_log.jsonl", "val_metrics.json") if (output_dir / n).exists()]
            if leftovers:
                raise RuntimeError(f"{run_label} có file dở nhưng thiếu last.pt: {leftovers}")
            queue.set_postfix_str(f"Trạng thái: đang train mới {run_label}")
        command = [sys.executable, "-u", "-m", "experiments.run_experiment", "--run", run_name, "--seed", str(seed), "--epochs", str(protocol["epochs"]), "--batch-size", str(protocol["batch_size"]), "--eval-batch-size", str(protocol["eval_batch_size"]), "--grad-accum-steps", str(protocol["grad_accum_steps"]), "--num-workers", str(protocol["num_workers"]), "--pad-to-multiple-of", str(protocol["pad_to_multiple_of"]), "--output-dir", str(output_dir)]
        if resume:
            command.append("--resume")
        metadata = {"schema_version": 2, "experiment_name": TEAM_CONFIG["experiment_name"], "member_id": MEMBER_ID, "role": "trainer", "run": run_name, "seed": seed, "status": "running", "resumed": resume, "resume_count": int(previous.get("resume_count", 0)) + int(resume), "started_at_utc": previous.get("started_at_utc", utc_now()), "updated_at_utc": utc_now(), "source_archive_sha256": SOURCE_SHA256, "config_sha256": CONFIG_SHA256, "command": command, "checkpoint_policy": protocol["checkpoint_policy"]}
        write_json(metadata_path, metadata)
        print(f"\\nBắt đầu {run_label}; đang nạp model và chuẩn bị DataLoader...", flush=True)
        try:
            run_with_live_output(command, output_dir, queue)
            assert summary_path.exists() and last_path.exists() and (output_dir / "best.pt").exists()
            metadata.update(status="complete", completed_at_utc=utc_now(), run_summary_sha256=sha256_file(summary_path))
            write_json(metadata_path, metadata)
        except BaseException as error:
            metadata.update(status="interrupted_or_failed", updated_at_utc=utc_now(), error=repr(error))
            write_json(metadata_path, metadata)
            raise
        queue.update(1)
    queue.set_postfix_str(f"Trạng thái: hoàn tất {len(jobs)}/{len(jobs)}", refresh=True)
print("Đã hoàn thành hàng đợi. last.pt và best.pt đều được giữ trên Drive.")
"""),
        markdown("## 7. Tạo gói kết quả gửi Member 4"),
        code("""from tqdm.auto import tqdm

bundle = EXPORT_ROOT / f"member_{MEMBER_ID}_seed{seed}_summaries.zip"
included = []
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    with tqdm(jobs, desc="Đóng gói kết quả", unit="model") as progress:
      for run_name in progress:
        progress.set_postfix_str(f"Trạng thái: đang đóng gói {run_name}")
        run_label = f"{run_name}_seed{seed}"
        run_dir = RESULTS_ROOT / run_label
        if not (run_dir / "run_summary.json").exists():
            continue
        for filename in ("run_summary.json", "team_job_metadata.json", "val_metrics.json", "train_log.jsonl"):
            source = run_dir / filename
            if source.exists():
                archive.write(source, arcname=f"{run_label}/{filename}")
        included.append(run_label)
      progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
assert len(included) == len(jobs), f"Mới hoàn thành {len(included)}/{len(jobs)} model"
print("Đã tạo:", bundle)
print("Gửi file ZIP này cho Member 4. Checkpoint vẫn nằm trên Drive của bạn.")
"""),
        markdown("## 8. Trạng thái cuối"),
        code("""from tqdm.auto import tqdm

rows = []
with tqdm(jobs, desc="Kiểm tra artifact", unit="model") as progress:
    for run_name in progress:
        progress.set_postfix_str(f"Trạng thái: đang kiểm tra {run_name}")
        run_dir = RESULTS_ROOT / f"{run_name}_seed{seed}"
        rows.append({"model": run_name, "summary": (run_dir / "run_summary.json").exists(), "best_checkpoint": (run_dir / "best.pt").exists(), "latest_checkpoint": (run_dir / "last.pt").exists(), "training_log": (run_dir / "train_log.jsonl").exists()})
    progress.set_postfix_str("Trạng thái: hoàn tất", refresh=True)
display(pd.DataFrame(rows))
assert all(all(row[key] for key in ("summary", "best_checkpoint", "latest_checkpoint", "training_log")) for row in rows)
print("HOÀN TẤT: đủ log, best.pt, last.pt và summary cho 2 model được phân công.")
"""),
    ]


def build_notebook() -> dict:
    return {"cells": build_cells(), "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "toc_visible": True}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[colab-trainers] wrote {args.output} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
