"""Exercise all Colab cells with a local tiny BERT and synthetic data."""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from experiments.final_test_package import FinalTest, atomic_json, digest
from src import config, guard
from src.models.baselines import BertPooledClassifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path):
    backbone = tmp_path / "tiny_bert"
    backbone.mkdir()
    (backbone / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nnews\nworld\nsports\nbusiness\ntech\n")
    BertTokenizerFast(vocab_file=str(backbone / "vocab.txt")).save_pretrained(backbone)
    BertModel(BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=24)).save_pretrained(backbone)
    root = tmp_path / "drive/LDTF_4_MEMBERS"
    c = json.loads((ROOT / "LDTF_4_MEMBERS/final_test/config.json").read_text(encoding="utf-8"))
    c.update(model_name=str(backbone), expected_test_rows=8, eval_batch_size=3)
    checkpoint = root / c["checkpoint"]
    checkpoint.parent.mkdir(parents=True)
    model = BertPooledClassifier(model_name=str(backbone))
    state = dict(architecture={"class_name": c["model_class"], "model_name": str(backbone),
        "pooling": "cls", "num_classes": 4, "backbone_frozen": False}, model_state_dict=model.state_dict(),
        seed=42, epoch=3, run_id=c["run_id"], data_signature={"fixture": "synthetic"},
        best_metrics={"f1_macro": c["expected_validation_f1"]})
    torch.save(state, checkpoint)
    pd.DataFrame({"text": ["news world", "news sports", "news business", "news tech"] * 2,
                  "label": [0, 1, 2, 3] * 2}).to_parquet(root / c["test_file"])
    c.update(checkpoint_sha256=digest(checkpoint), test_sha256=digest(root / c["test_file"]))
    atomic_json(root / "final_test/config.json", c)
    evidence = root / c["selection_evidence"]
    summary = {"run_id": c["run_id"], "base_run": "B2_bert_finetuned_cls", "data_signature": state["data_signature"]}
    atomic_json(evidence / "run_summary.json", summary)
    atomic_json(evidence / "team_job_metadata.json", {"run": summary["base_run"], "seed": 42,
        "status": "complete", "run_summary_sha256": digest(evidence / "run_summary.json")})
    atomic_json(evidence / "protocol_violations.json", [])
    pd.DataFrame({"rank": range(1, 7), "run": [summary["base_run"], "A11", "A0", "A3", "A1", "A4"],
        "seed": [42]*6, "protocol_ok": [True]*6, "val_f1_macro": [c["expected_validation_f1"]]*6}).to_csv(evidence / "validation_ranking.csv", index=False)
    import shutil
    shutil.copytree(ROOT / "LDTF_4_MEMBERS/final_test/source", root / "final_test/source")
    return root


def test_all_colab_cells_and_rerun(package, monkeypatch):
    monkeypatch.setenv("LDTF_COLAB_SIMULATION_ROOT", str(package))
    monkeypatch.delenv(guard.UNLOCK_ENV_VAR, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_test = config.PROCESSED_TEST
    notebook = json.loads((ROOT / "notebooks/colab_final_test_B2.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 9
    namespace = {}
    for cell in cells:
        exec(compile("".join(cell["source"]), "<colab-cell>", "exec"), namespace)
    workflow = namespace["workflow"]
    assert config.PROCESSED_TEST == original_test
    assert not guard.is_unlocked()
    with np.load(workflow.pred_path) as predictions:
        assert predictions["logits"].shape == (8, 4)
        assert predictions["labels"].tolist() == [0, 1, 2, 3]*2
    ledger = workflow.out / "logs/official_test_access.jsonl"
    assert len(ledger.read_text().splitlines()) == 1
    assert json.loads(workflow.state_path.read_text())["status"] == "complete"
    assert zipfile.is_zipfile(namespace["bundle"])
    assert workflow.out == package / "final_test/colab_runs/B2_colab_detailed_01"
    with zipfile.ZipFile(namespace["bundle"]) as archive:
        for name in ("reports/DETAILED_REPORT.html", "metrics/extended_metrics.json",
                     "figures/calibration.png", "predictions/all_predictions.csv"):
            assert name in archive.namelist()
        manifest = json.loads(archive.read("reports/artifact_manifest.json"))
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    for cell in cells[1:]:
        exec(compile("".join(cell["source"]), "<colab-rerun>", "exec"), namespace)
    assert len(ledger.read_text().splitlines()) == 1
    with workflow.pred_path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        workflow.cached()


def test_bad_hash_rejected_before_test(package):
    w = FinalTest(package)
    w.cfg["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        w.validate()
    assert not (w.out / "logs/official_test_access.jsonl").exists()


def test_wrong_seed_rejected(package):
    w = FinalTest(package)
    w.validate()
    w.cfg["seed"] = 1337
    with pytest.raises(ValueError, match="Checkpoint"):
        w.check_checkpoint()


def test_windows_summary_line_endings(package):
    w = FinalTest(package)
    path = w.evidence / "run_summary.json"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    w.validate()
    w.check_checkpoint()
    assert w.checked


def test_local_gpu_required_and_separate_output(package, monkeypatch):
    from experiments.local_test_report import LocalTest
    w = LocalTest(package, 'unit_gpu', 8)
    assert w.out.name == 'unit_gpu'
    assert w.cfg['eval_batch_size'] == 8
    w.validate(); w.check_checkpoint()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        w.prepare()


def test_extended_report_perfect_predictions(package):
    from experiments.local_test_report import LocalTest, extended_report, export_local
    w = LocalTest(package, 'unit_report')
    labels = np.arange(8) % 4
    logits = np.eye(4)[labels] * 30
    np.savez_compressed(w.pred_path, logits=logits, labels=labels, predictions=labels,
        texts=np.array(['<script>example</script>']*8), row_index=np.arange(8))
    atomic_json(w.state_path, {'identity': w.identity, 'status': 'predicted',
                              'predictions_sha256': digest(w.pred_path)})
    w.analyze()
    stats = extended_report(w, 100)
    assert stats['accuracy_ci95'] == [1.0, 1.0]
    assert stats['macro_f1_ci95'][0] <= stats['macro_f1_ci95'][1] <= 1
    assert stats['multiclass_brier_sum'] < 1e-10
    assert stats['incorrect'] == 0
    bundle = export_local(w)
    with zipfile.ZipFile(bundle) as archive:
        assert 'reports/DETAILED_REPORT.html' in archive.namelist()
        manifest = json.loads(archive.read('reports/artifact_manifest.json'))
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected


def test_interruption_restores_guard(package, monkeypatch):
    w = FinalTest(package)
    w.validate(); w.check_checkpoint(); w.prepare()
    monkeypatch.delenv(guard.UNLOCK_ENV_VAR, raising=False)
    original = config.PROCESSED_TEST
    original_forward = w.model.forward
    def fail(*args, **kwargs):
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(w.model, "forward", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        w.evaluate()
    assert not guard.is_unlocked() and config.PROCESSED_TEST == original
    assert json.loads(w.state_path.read_text())["status"] == "interrupted_or_failed"
    assert not w.pred_path.exists()
    monkeypatch.setattr(w.model, "forward", original_forward)
    w.evaluate()
    assert w.cached()
    assert len((w.out / "logs/official_test_access.jsonl").read_text().splitlines()) == 2
