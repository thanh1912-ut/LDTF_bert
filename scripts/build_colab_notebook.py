"""Generate the Colab notebook.

Usage::

    python -m scripts.build_colab_notebook
    python -m scripts.build_colab_notebook --output notebooks/colab_ldtf_bert.ipynb

The notebook is a build artifact: edit this generator, not the .ipynb.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config

REPO_PLACEHOLDER = "https://github.com/thanh1912-ut/LDTF_bert.git"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown(
            "# LDTF-BERT on AG News — Colab\n"
            "\n"
            "Label-Directed Token and Depth Fusion over BERT layers.\n"
            "\n"
            "**Before you start:** set **Runtime -> Change runtime type -> GPU** (T4 is enough).\n"
            "\n"
            "This is the single-GPU Colab notebook. For Kaggle T4 x2, use the "
            "`python -m torch.distributed.run` commands in `README.md`.\n"
            "\n"
            "This notebook trains and validates only. The official test split stays sealed\n"
            "until the final locked evaluation at the end, which is opt-in and logged.\n"
        )
    )

    cells.append(markdown("## 1. Check the GPU"))
    cells.append(
        code(
            "!nvidia-smi\n"
            "\n"
            "import torch\n"
            "print('torch', torch.__version__)\n"
            "print('cuda available:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print('device:', torch.cuda.get_device_name(0))\n"
            "    print('bf16 supported:', torch.cuda.is_bf16_supported())\n"
        )
    )

    cells.append(
        markdown(
            "## 2. Get the code and data\n"
            "\n"
            "Pick **one** of the two options below.\n"
            "\n"
            "**Option A — Google Drive (recommended, no git needed).** On your own\n"
            "machine, from the project root, build a single archive:\n"
            "\n"
            "```bash\n"
            "tar --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs' \\\n"
            "    --exclude='src/data' -czf ldtf_bert.tar.gz \\\n"
            "    src experiments scripts tests docs requirements.txt README.md data/processed\n"
            "```\n"
            "\n"
            "That is roughly 100 MB (0.6 MB of code plus the three parquet splits).\n"
            "Upload `ldtf_bert.tar.gz` to your Drive, then set `USE_DRIVE = True` below.\n"
            "\n"
            "**Option B — git clone.** Only works if the repository is pushed somewhere\n"
            "Colab can reach, and the parquet files must be included or fetched separately.\n"
            "\n"
            "Either way, `PROJECT` must end up being the directory that contains `src/`."
        )
    )
    cells.append(
        code(
            "import os, sys\n"
            "from pathlib import Path\n"
            "\n"
            "USE_DRIVE = True                     # False to clone from git instead\n"
            "ARCHIVE   = '/content/drive/MyDrive/ldtf_bert.tar.gz'\n"
            f"REPO_URL  = {REPO_PLACEHOLDER!r}\n"
            "PROJECT   = Path('/content/HocSau_LDTF_BERT')\n"
            "\n"
            "if USE_DRIVE:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    assert Path(ARCHIVE).exists(), f'archive not found at {ARCHIVE}'\n"
            "    PROJECT.mkdir(parents=True, exist_ok=True)\n"
            "    !tar -xzf $ARCHIVE -C $PROJECT\n"
            "else:\n"
            "    if not PROJECT.exists():\n"
            "        !git clone $REPO_URL $PROJECT\n"
            "\n"
            "assert (PROJECT / 'src').is_dir(), f'src/ not found under {PROJECT}'\n"
            "os.chdir(PROJECT)\n"
            "sys.path.insert(0, str(PROJECT))\n"
            "print('working directory:', Path.cwd())\n"
        )
    )

    cells.append(markdown("## 3. Install dependencies"))
    cells.append(
        code("!pip install -q -r requirements.txt\nprint('dependencies installed')\n")
    )

    cells.append(
        markdown(
            "## 4. Check the data\n"
            "\n"
            "Expected under `data/processed/`: `research_train.parquet`,\n"
            "`research_validation.parquet`, `research_test.parquet`.\n"
            "The test file is present but sealed; nothing below reads it."
        )
    )
    cells.append(
        code(
            "from src import config\n"
            "from src.dataset import load_split\n"
            "\n"
            "for name, path in [('train', config.PROCESSED_TRAIN), ('validation', config.PROCESSED_VAL)]:\n"
            "    frame = load_split(path)\n"
            "    counts = frame[config.LABEL_COLUMN].value_counts().sort_index().tolist()\n"
            "    print(f'{name:<11} {len(frame):>7,} rows   per-class {counts}')\n"
            "\n"
            "print('\\nofficial test split:', config.PROCESSED_TEST.name,\n"
            "      '(sealed — see src/guard.py)')\n"
        )
    )

    cells.append(
        markdown(
            "## 5. Smoke test\n"
            "\n"
            "Drives all 22 configurations through forward, backward, optimizer coverage,\n"
            "gradient audit, checkpointing, resume and evaluation on a tiny synthetic model.\n"
            "Takes about a minute and catches an environment problem before a long run."
        )
    )
    cells.append(code("!python -m scripts.smoke_test 2>&1 | tail -n 25\n"))

    cells.append(
        markdown(
            "## 6. Quick sanity run\n"
            "\n"
            "A short run on a small subset to confirm the live display works and loss moves.\n"
            "These are **debug numbers, not results** — the summary records\n"
            "`is_debug_subset: true`."
        )
    )
    cells.append(
        code(
            "!python -m experiments.run_experiment --run A0 --limit-train-rows 2000 --epochs 1 --batch-size 32\n"
        )
    )

    cells.append(
        markdown(
            "## 7. Train one configuration\n"
            "\n"
            "The live display is YOLO-style: a per-batch bar with GPU memory, running loss\n"
            "and both learning rates, then a per-class precision/recall/F1 table after each\n"
            "epoch, then a confusion matrix at the end.\n"
            "\n"
            "Run ids: `A0`–`A14`, and `B1_bert_frozen_cls`, `B2_bert_finetuned_cls`,\n"
            "`B3_bert_frozen_mean_pool`, `B4_bert_scalar_mix`, `B5_token_attention_only`,\n"
            "`B6_full_ldtf_frozen`, `B7_full_ldtf_finetuned`.\n"
            "\n"
            "Add `--resume` to continue from `last.pt` if Colab disconnects."
        )
    )
    cells.append(
        code(
            "RUN  = 'A0'\n"
            "SEED = 42\n"
            "\n"
            "!python -m experiments.run_experiment --run $RUN --seed $SEED --batch-size 32\n"
        )
    )

    cells.append(
        markdown(
            "## 8. Train a suite with a live leaderboard\n"
            "\n"
            "Presets: `core` (A0, A1, A3, A4, A11 — the comparisons that decide the\n"
            "question), `baselines`, `ablations`, `all`.\n"
            "\n"
            "The leaderboard redraws after every run, sorted by validation macro F1.\n"
            "Completed runs are skipped on restart, so you can just rerun this cell after\n"
            "a disconnect."
        )
    )
    cells.append(
        code(
            "!python -m experiments.run_suite --preset core --seed 42 --continue-on-error\n"
        )
    )

    cells.append(
        markdown(
            "### Watch the disk\n"
            "\n"
            "Each run writes `best.pt` (~440 MB) and `last.pt` (~1.3 GB). Colab gives you\n"
            "roughly 80 GB, so the full 19-configuration suite will not fit. `last.pt` is\n"
            "only needed to resume, so delete it for runs that already finished."
        )
    )
    cells.append(
        code(
            "!df -h /content | tail -1\n"
            "\n"
            "from pathlib import Path\n"
            "\n"
            "DELETE_LAST = False   # True to reclaim space from completed runs\n"
            "\n"
            "for summary in sorted(Path('outputs').glob('*/run_summary.json')):\n"
            "    directory = summary.parent\n"
            "    last = directory / 'last.pt'\n"
            "    size = last.stat().st_size / 2**30 if last.exists() else 0.0\n"
            "    if DELETE_LAST and last.exists():\n"
            "        last.unlink()\n"
            "        print(f'{directory.name:<32} freed {size:.2f} GB')\n"
            "    else:\n"
            "        print(f'{directory.name:<32} last.pt {size:.2f} GB')\n"
        )
    )

    cells.append(
        markdown(
            "### The classical baseline (B0)\n"
            "TF-IDF + logistic regression. `C` is tuned on validation only."
        )
    )
    cells.append(code("!python -m experiments.run_tfidf_baseline\n"))

    cells.append(
        markdown(
            "## 9. Inspect what the routers learned\n"
            "\n"
            "The scientific claim is that different classes attend to different layers.\n"
            "If these rows are near-identical, LDTF has collapsed to a scalar mix and the\n"
            "contribution is not real — regardless of accuracy."
        )
    )
    cells.append(
        code(
            "import torch\n"
            "from src import config\n"
            "from src.dataset import build_dataloaders\n"
            "from src.evaluate import load_model_from_checkpoint\n"
            "from src.models import LdtfBert\n"
            "from src.train import forward_kwargs, move_batch_to_device\n"
            "from src.utils import get_device\n"
            "\n"
            "RUN_DIR = config.experiment_output_dir('A0_seed42')\n"
            "model, _ = load_model_from_checkpoint(RUN_DIR / 'best.pt')\n"
            "device = get_device(); model.to(device).eval()\n"
            "\n"
            "tokenizer = LdtfBert.build_tokenizer(config.MODEL_NAME)\n"
            "loaders = build_dataloaders(tokenizer, batch_size=64)\n"
            "batch = move_batch_to_device(next(iter(loaders['validation'])), device)\n"
            "\n"
            "with torch.no_grad():\n"
            "    out = model(**forward_kwargs(batch), return_routing=True)\n"
            "depth = out['depth_attention'].float().mean(0).cpu()   # [C, L]\n"
            "\n"
            "print('Mean depth attention per class (rows sum to 1)\\n')\n"
            "print('class      ' + ''.join(f'L{i + 1:<5}' for i in range(depth.shape[1])))\n"
            "for index, name in enumerate(config.LABEL_NAMES):\n"
            "    print(f'{name:<11}' + ''.join(f'{v:<6.3f}' for v in depth[index].tolist()))\n"
            "\n"
            "spread = (depth.max(0).values - depth.min(0).values).max().item()\n"
            "print(f'\\nlargest between-class gap on any layer: {spread:.4f}')\n"
            "print('near zero would mean the classes agree, i.e. no label conditioning in practice')\n"
        )
    )
    cells.append(
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "figure, axis = plt.subplots(figsize=(9, 3.2))\n"
            "image = axis.imshow(depth.numpy(), aspect='auto', cmap='viridis')\n"
            "axis.set_yticks(range(len(config.LABEL_NAMES)))\n"
            "axis.set_yticklabels(config.LABEL_NAMES)\n"
            "axis.set_xticks(range(depth.shape[1]))\n"
            "axis.set_xticklabels([f'L{i + 1}' for i in range(depth.shape[1])])\n"
            "axis.set_title('Depth attention per class')\n"
            "figure.colorbar(image)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        markdown(
            "## 10. Export the result tables\n"
            "\n"
            "Rows that have not been run show `PENDING`. Nothing is estimated or back-filled."
        )
    )
    cells.append(
        code(
            "!python -m experiments.export_tables --with-params\n"
            "\n"
            "from pathlib import Path\n"
            "from IPython.display import Markdown, display\n"
            "\n"
            "for name in ('baseline_table.md', 'ablation_table.md'):\n"
            "    display(Markdown((Path('reports/tables') / name).read_text()))\n"
        )
    )

    cells.append(
        markdown(
            "## 11. Compare two runs properly\n"
            "\n"
            "At n = 7,600 a single run's standard error is about 0.27 pp, so a gap under\n"
            "roughly 0.75 pp is not distinguishable from noise. Use a paired test."
        )
    )
    cells.append(
        code(
            "import numpy as np\n"
            "from src.metrics import bootstrap_accuracy_difference, mcnemar_exact\n"
            "\n"
            "# Requires step 12 to have written test_predictions.npz for both runs.\n"
            "A, B = 'A0_seed42', 'A1_seed42'\n"
            "try:\n"
            "    left  = np.load(config.experiment_output_dir(A) / 'test_predictions.npz')\n"
            "    right = np.load(config.experiment_output_dir(B) / 'test_predictions.npz')\n"
            "    labels = left['labels']\n"
            "    test = mcnemar_exact(left['predictions'], right['predictions'], labels)\n"
            "    ci = bootstrap_accuracy_difference(left['predictions'], right['predictions'], labels)\n"
            "    print(f'{A} vs {B}')\n"
            "    print(f\"  McNemar b={test['b']:.0f} c={test['c']:.0f} p={test['p_value']:.4f}\")\n"
            "    print(f\"  accuracy difference {ci['mean_difference']:+.4f} \"\n"
            "          f\"95% CI [{ci['ci_lower_95']:+.4f}, {ci['ci_upper_95']:+.4f}]\")\n"
            "    if ci['ci_lower_95'] <= 0 <= ci['ci_upper_95']:\n"
            "        print('  the interval crosses zero: report this as a null result')\n"
            "except FileNotFoundError:\n"
            "    print('Run step 12 first to produce test predictions.')\n"
        )
    )

    cells.append(
        markdown(
            "## 12. Locked final evaluation (run once, at the very end)\n"
            "\n"
            "**Stop.** Only run this after every architecture, hyper-parameter, epoch and\n"
            "seed decision is final. Using it earlier invalidates the results.\n"
            "\n"
            "Access requires an explicit unlock token, is refused while training is active,\n"
            "and appends a hash-stamped record to `reports/official_test_access.jsonl`. The\n"
            "`access_index` in that ledger is how many times the test set has ever been\n"
            "touched, and it belongs in the write-up."
        )
    )
    cells.append(
        code(
            "# Uncomment to unseal. Do this once.\n"
            "#\n"
            "# import os\n"
            "# os.environ['LDTF_ALLOW_OFFICIAL_TEST'] = 'I_AM_REPORTING_FINAL_RESULTS'\n"
            "# !python -m experiments.final_eval --run A0_seed42 --run A1_seed42 --reason 'final reported numbers'\n"
            "print('sealed — uncomment above only when all model selection is complete')\n"
        )
    )

    cells.append(
        markdown(
            "## 13. Save results to Drive\n"
            "\n"
            "Colab wipes `/content` when the runtime recycles. Copy `outputs/` and\n"
            "`reports/` if you are not already working inside Drive."
        )
    )
    cells.append(
        code(
            "SAVE_TO_DRIVE = False\n"
            "\n"
            "if SAVE_TO_DRIVE:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive', force_remount=True)\n"
            "    destination = '/content/drive/MyDrive/ldtf_bert_results'\n"
            "    !mkdir -p $destination\n"
            "    !cp -r outputs $destination/\n"
            "    !cp -r reports $destination/\n"
            "    print('saved to', destination)\n"
            "else:\n"
            "    print('set SAVE_TO_DRIVE = True to copy outputs/ and reports/ to Drive')\n"
        )
    )

    return cells


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Colab notebook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.ROOT_DIR / "notebooks" / "colab_ldtf_bert.ipynb",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    args.output.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[colab] wrote {args.output} ({len(notebook['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
