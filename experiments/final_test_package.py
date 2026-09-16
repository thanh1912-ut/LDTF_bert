"""Drive-backed final evaluation workflow used by the Member 4 notebook."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src import config, guard
from src.metrics import compute_classification_metrics
from src.registry import build_model_from_architecture
from src.train import forward_kwargs, move_batch_to_device


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame, path, **kwargs):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, **kwargs)
    os.replace(temporary, path)


def digest(path, show=False):
    path = Path(path)
    result = hashlib.sha256()
    with path.open("rb") as stream, tqdm(total=path.stat().st_size, desc=path.name,
            unit="B", unit_scale=True, disable=not show, leave=False) as bar:
        bar.set_postfix_str("Đang kiểm tra SHA-256")
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
            bar.update(len(block))
    return result.hexdigest()


@contextmanager
def activity(label):
    """Keep a visible heartbeat during operations without measurable progress."""
    stop = threading.Event()
    started = time.monotonic()
    with tqdm(total=1, desc=label, unit="bước") as bar:
        def heartbeat():
            while not stop.wait(1):
                bar.set_postfix_str(f"Đang xử lý | {time.monotonic() - started:.0f}s")
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            bar.set_postfix_str("Đang xử lý")
            yield
        except BaseException:
            bar.set_postfix_str("Lỗi hoặc bị ngắt; xem thông báo bên dưới")
            raise
        else:
            bar.update(1)
            bar.set_postfix_str("Hoàn tất")
        finally:
            stop.set()
            worker.join()


class FinalTest:
    def __init__(self, root, *, overrides=None):
        self.root = Path(root).resolve()
        self.cfg = read_json(self.root / "final_test/config.json")
        if overrides:
            self.cfg.update(overrides)
        if not self.cfg.get("evaluation_enabled"):
            raise ValueError("Final-test chưa được bật trong config.")
        self.checkpoint = self.path(self.cfg["checkpoint"])
        self.test_file = self.path(self.cfg["test_file"])
        self.evidence = self.path(self.cfg["selection_evidence"])
        self.out = self.path(self.cfg["output_dir"])
        for name in ("metrics", "predictions", "figures", "logs", "reports"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.identity = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        self.pred_path = self.out / "predictions/test_predictions.npz"
        self.state_path = self.out / "logs/evaluation_state.json"
        self.model = self.tokenizer = None
        self.validated = self.checked = False

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Đường dẫn ngoài thư mục bàn giao: {relative}")
        return path

    def log(self, message):
        with (self.out / "logs/console.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        print(message, flush=True)

    def validate(self):
        with tqdm(total=3, desc="Kiểm tra đầu vào", unit="bước") as bar:
            for key, path in (("checkpoint", self.checkpoint), ("test", self.test_file)):
                bar.set_postfix_str(f"SHA-256: {path.name}")
                if digest(path, show=True) != self.cfg[f"{key}_sha256"]:
                    raise ValueError(f"Sai checksum: {path}")
                bar.update(1)
            bar.set_postfix_str("Đối chiếu lựa chọn validation")
            self.summary = read_json(self.evidence / "run_summary.json")
            metadata = read_json(self.evidence / "team_job_metadata.json")
            summary_bytes = (self.evidence / "run_summary.json").read_bytes()
            # Git on Windows may convert only line endings in the JSON artifact.
            summary_hashes = {hashlib.sha256(summary_bytes).hexdigest(),
                hashlib.sha256(summary_bytes.replace(b"\r\n", b"\n")).hexdigest()}
            if metadata["run_summary_sha256"] not in summary_hashes:
                raise ValueError("Summary không khớp metadata (kể cả chuẩn hóa CRLF/LF).")
            ranking = pd.read_csv(self.evidence / "validation_ranking.csv")
            winner = ranking.sort_values("rank").iloc[0]
            if (read_json(self.evidence / "protocol_violations.json") or len(ranking) != 6
                    or not ranking["protocol_ok"].astype(str).str.lower().eq("true").all()
                    or not ranking["seed"].eq(self.cfg["seed"]).all()
                    or metadata["status"] != "complete"
                    or metadata["run"] != self.summary["base_run"]
                    or metadata["seed"] != self.cfg["seed"]
                    or winner["run"] != self.summary["base_run"]
                    or self.summary["run_id"] != self.cfg["run_id"]
                    or abs(winner["val_f1_macro"] - self.cfg["expected_validation_f1"]) > 1e-10):
                raise ValueError("Bằng chứng chọn model không hợp lệ.")
            if (self.cfg["max_length"] != 128 or self.cfg["text_encoding"] != "single"
                    or self.cfg["label_names"] != list(config.LABEL_NAMES)):
                raise ValueError("Encoding hoặc label mapping khác protocol.")
            self.validated = True
            bar.update(1)
            bar.set_postfix_str("Đủ đầu vào; chưa đọc nội dung test")
        self.log("Đầu vào và lựa chọn validation hợp lệ.")

    def check_checkpoint(self):
        if not self.validated:
            raise RuntimeError("Chạy cell kiểm tra đầu vào trước.")
        with activity("Đọc metadata checkpoint"):
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                self.state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            state, c = self.state, self.cfg
            arch = state["architecture"]
            expected = {"class_name": c["model_class"], "model_name": c["model_name"],
                        "pooling": c["pooling"], "num_classes": 4, "backbone_frozen": False}
            if (any(arch.get(k) != v for k, v in expected.items())
                    or state["seed"] != c["seed"] or state["run_id"] != c["run_id"]
                    or state["epoch"] != c["expected_best_epoch"]
                    or state["data_signature"] != self.summary["data_signature"]
                    or abs(state["best_metrics"]["f1_macro"] - c["expected_validation_f1"]) > 1e-10):
                raise ValueError("Checkpoint khác model/seed/epoch/dữ liệu đã chọn.")
            self.checked = True
        self.cached()
        self.log(f"Đã xác minh {c['run_id']}, epoch {state['epoch']}.")

    def cached(self):
        if not self.state_path.exists():
            if self.pred_path.exists():
                raise ValueError("Có dự đoán nhưng thiếu state; cần kiểm tra trước khi chạy lại.")
            return False
        previous = read_json(self.state_path)
        if previous["identity"] != self.identity:
            raise ValueError("Cấu hình khác lần chạy trước; không ghi đè kết quả.")
        if previous["status"] in ("predicted", "complete"):
            if not self.pred_path.exists() or digest(self.pred_path) != previous["predictions_sha256"]:
                raise ValueError("Dự đoán đã lưu bị thiếu hoặc sai checksum.")
            return True
        return False

    def prepare(self):
        if not self.checked:
            raise RuntimeError("Chưa xác minh checkpoint.")
        with activity("Chuẩn bị model và tokenizer"):
            if self.cached():
                self.log("Đã có dự đoán hợp lệ; dùng lại để tạo báo cáo.")
                return
            torch.manual_seed(self.cfg["seed"])
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"Thiết bị: {self.device}; inference float32, batch {self.cfg['eval_batch_size']}.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name"])
            self.model = build_model_from_architecture(self.state["architecture"])
            self.model.load_state_dict(self.state["model_state_dict"], strict=True)
            self.model.to(self.device).eval()

    def evaluate(self):
        if not self.checked:
            raise RuntimeError("Chưa xác minh checkpoint.")
        with FileLock(str(self.out / "logs/evaluation.lock"), timeout=0):
            if self.cached():
                with activity("Đọc trạng thái dự đoán"):
                    self.log("Dùng dự đoán đã lưu; không truy cập lại tập test.")
                return
            if self.model is None:
                raise RuntimeError("Chạy cell chuẩn bị model trước.")
            old_test, old_ledger = config.PROCESSED_TEST, config.TEST_ACCESS_LEDGER
            old_unlock = os.environ.get(guard.UNLOCK_ENV_VAR)
            config.PROCESSED_TEST = self.test_file
            config.TEST_ACCESS_LEDGER = self.out / "logs/official_test_access.jsonl"
            run_state = {"identity": self.identity, "status": "running", "started_at": time.time()}
            atomic_json(self.state_path, run_state)
            try:
                os.environ[guard.UNLOCK_ENV_VAR] = guard.UNLOCK_TOKEN
                with activity("Mở tập test và tạo DataLoader"):
                    loader = guard.official_test_loader(self.tokenizer, reason="Final B2 selected on validation",
                        run_id=self.cfg["run_id"], batch_size=self.cfg["eval_batch_size"],
                        checkpoint_path=self.checkpoint)
                    if len(loader.dataset) != self.cfg["expected_test_rows"]:
                        raise ValueError("Số mẫu test không khớp config.")
                logits, labels = [], []
                with torch.inference_mode(), tqdm(loader, desc="Đánh giá B2", unit="batch") as bar:
                    for batch in bar:
                        batch = move_batch_to_device(batch, self.device)
                        value = self.model(**forward_kwargs(batch))["logits"]
                        if not torch.isfinite(value).all():
                            raise ValueError("Logits chứa NaN hoặc Inf.")
                        logits.append(value.float().cpu().numpy())
                        labels.append(batch["labels"].cpu().numpy())
                        bar.set_postfix_str(f"Đang dự đoán | {sum(len(x) for x in labels)}/{len(loader.dataset)} mẫu")
                    bar.set_postfix_str("Dự đoán hoàn tất")
                with activity("Lưu dự đoán lên Drive"):
                    temporary = self.pred_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        scores, truth = np.concatenate(logits), np.concatenate(labels)
                        np.savez_compressed(stream, logits=scores, labels=truth,
                            predictions=scores.argmax(-1), row_index=np.arange(len(truth)),
                            texts=np.asarray(loader.dataset.first_texts, dtype=str))
                    os.replace(temporary, self.pred_path)
                    run_state.update(status="predicted", predictions_sha256=digest(self.pred_path))
                    atomic_json(self.state_path, run_state)
                self.log(f"Đã lưu {len(truth)} dự đoán, thời gian {time.time()-run_state['started_at']:.1f}s.")
            except BaseException as error:
                run_state.update(status="interrupted_or_failed", error=repr(error))
                atomic_json(self.state_path, run_state)
                self.log(f"Đánh giá bị ngắt/lỗi: {error!r}")
                raise
            finally:
                config.PROCESSED_TEST, config.TEST_ACCESS_LEDGER = old_test, old_ledger
                if old_unlock is None:
                    os.environ.pop(guard.UNLOCK_ENV_VAR, None)
                else:
                    os.environ[guard.UNLOCK_ENV_VAR] = old_unlock

    def analyze(self):
        if not self.cached():
            raise RuntimeError("Chưa có dự đoán hoàn chỉnh.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        with tqdm(total=4, desc="Phân tích test", unit="bước") as bar:
            bar.set_postfix_str("Tính chỉ số từ dự đoán đã lưu")
            with np.load(self.pred_path, allow_pickle=False) as data:
                logits, labels, predictions = data["logits"], data["labels"], data["predictions"]
                texts, rows = data["texts"], data["row_index"]
            metrics = compute_classification_metrics(logits, labels, label_names=self.cfg["label_names"])
            metrics.update(loss=float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),
                           torch.from_numpy(labels).long())), run_id=self.cfg["run_id"],
                           checkpoint_sha256=self.cfg["checkpoint_sha256"], test_sha256=self.cfg["test_sha256"],
                           checkpoint_epoch=self.cfg["expected_best_epoch"], inference_dtype="float32")
            atomic_json(self.out / "metrics/test_metrics.json", metrics)
            atomic_csv(pd.DataFrame(metrics["per_class"]).T.rename_axis("label"), self.out / "metrics/per_class_metrics.csv")
            bar.update(1)
            bar.set_postfix_str("Xuất các mẫu dự đoán sai")
            wrong = predictions != labels
            atomic_csv(pd.DataFrame({"row_index": rows[wrong], "text": texts[wrong],
                          "label": labels[wrong], "prediction": predictions[wrong]}),
                          self.out / "predictions/errors.csv", index=False)
            bar.update(1)
            bar.set_postfix_str("Vẽ confusion matrix")
            fig, ax = plt.subplots(figsize=(8, 7))
            ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]),
                display_labels=self.cfg["label_names"]).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title("B2 seed 42 - Test"); fig.tight_layout()
            fig.savefig(self.out / "figures/confusion_matrix.png", dpi=160)
            plt.close(fig)
            bar.update(1)
            bar.set_postfix_str("Vẽ F1 theo nhãn")
            fig, ax = plt.subplots(figsize=(8, 4))
            names = self.cfg["label_names"]
            ax.bar(names, [metrics["per_class"][n]["f1"] for n in names], color=["#26786e", "#4875a1", "#b55561", "#767b35"])
            ax.set_ylim(0, 1); ax.set_ylabel("F1"); fig.tight_layout()
            fig.savefig(self.out / "figures/per_class_f1.png", dpi=160)
            plt.close(fig)
            bar.update(1); bar.set_postfix_str("Hoàn tất")
        self.log(f"Test: accuracy={metrics['accuracy']:.6f}, Macro F1={metrics['f1_macro']:.6f}.")
        return metrics

    def export(self):
        if not self.cached():
            raise RuntimeError("Chưa có dự đoán hoàn chỉnh.")
        required = ["metrics/test_metrics.json", "metrics/per_class_metrics.csv",
                    "predictions/errors.csv", "figures/confusion_matrix.png", "figures/per_class_f1.png",
                    "logs/official_test_access.jsonl"]
        for name in required:
            if not (self.out / name).is_file():
                raise FileNotFoundError(f"Chưa đủ đầu ra; chạy cell 6: {name}")
        with tqdm(total=3, desc="Xuất báo cáo", unit="bước") as bar:
            bar.set_postfix_str("Viết báo cáo tiếng Việt")
            metrics = read_json(self.out / "metrics/test_metrics.json")
            report = (f"# Báo cáo final-test B2\n\n"
                f"- Model: {self.cfg['run_id']}, epoch {self.cfg['expected_best_epoch']}.\n"
                f"- Số mẫu test: {metrics['num_examples']}.\n"
                f"- Accuracy: {metrics['accuracy']:.6f}.\n"
                f"- Macro F1: {metrics['f1_macro']:.6f}.\n"
                f"- Loss: {metrics['loss']:.6f}.\n"
                f"- Macro F1 validation: {self.cfg['expected_validation_f1']:.6f}.\n"
                f"- Chênh lệch test - validation: {metrics['f1_macro']-self.cfg['expected_validation_f1']:+.6f}.\n\n"
                "Model được chọn bằng validation trước khi truy cập test. Chỉ đánh giá một seed; "
                "chưa đo độ ổn định qua nhiều seed. Test thuộc AG News, không đại diện đầy đủ "
                "cho BBC/HuffPost. Chênh lệch validation/test không tự chứng minh overfitting.\n\n"
                "Xem `../metrics/per_class_metrics.csv` để đối chiếu từng nhãn và "
                "`../predictions/errors.csv` để đọc mẫu sai; row_index là vị trí dòng 0-based "
                "trong test. Dự đoán NPZ chứa logits, labels, predictions, row_index, texts.\n\n"
                "![Confusion matrix](../figures/confusion_matrix.png)\n\n"
                "![F1 theo nhãn](../figures/per_class_f1.png)\n")
            target = self.out / "reports/FINAL_TEST_REPORT.md"
            temp = target.with_suffix(".tmp"); temp.write_text(report, encoding="utf-8"); os.replace(temp, target)
            bar.update(1)
            bar.set_postfix_str("Lập manifest checksum")
            files = [p for p in self.out.rglob("*") if p.is_file() and p.suffix not in (".tmp", ".lock")
                     and p.name not in (".gitkeep", "artifact_manifest.json", "evaluation_state.json")]
            atomic_json(self.out / "reports/artifact_manifest.json",
                {"identity": self.identity, "files": {p.relative_to(self.out).as_posix(): digest(p) for p in files}})
            bar.update(1)
            bar.set_postfix_str("Đóng gói ZIP kết quả")
            bundle = self.path(self.cfg["bundle_file"]) if "bundle_file" in self.cfg else self.root / "final_test" / (self.cfg["run_id"] + "_final_test.zip")
            bundle.parent.mkdir(parents=True, exist_ok=True)
            temporary = bundle.with_suffix(".tmp")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in files + [self.out / "reports/artifact_manifest.json"]:
                    archive.write(path, str(path.relative_to(self.out)))
            os.replace(temporary, bundle)
            state = read_json(self.state_path); state["status"] = "complete"
            atomic_json(self.state_path, state)
            bar.update(1); bar.set_postfix_str("Hoàn tất; báo cáo đã lưu trên Drive")
        print(bundle)
        return bundle
