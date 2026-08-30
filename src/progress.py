"""YOLO-style progress reporting for training runs.

Three layers of display, all optional and all side-effect free with respect to
the training result:

``BatchProgress``
    A per-batch bar in the Ultralytics/YOLO idiom: a fixed header, then one
    line per epoch showing GPU memory, running loss, learning rates and the
    batch rate.

``EpochTable``
    After each validation pass, a per-class table with precision, recall and F1
    plus an "all" summary row, mirroring YOLO's per-class mAP table.

``SuiteBoard``
    A live leaderboard across many runs, sorted by validation macro F1, redrawn
    as each run finishes.

Nothing here changes optimisation. Disable with ``--no-progress`` or by passing
``reporter=None``.
"""

from __future__ import annotations

import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from . import config

try:  # pragma: no cover - exercised implicitly
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def supports_color(stream=None) -> bool:
    """Return whether ANSI colour is safe to emit."""
    stream = stream or sys.stdout
    if not hasattr(stream, "isatty"):
        return False
    if stream.isatty():
        return True
    # Colab and Jupyter render ANSI even though isatty() is False.
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def _paint(text: str, colour: str, *, enabled: bool) -> str:
    return f"{colour}{text}{RESET}" if enabled else text


def format_duration(seconds: float) -> str:
    """Render a duration as ``1:23:45``, ``4:05`` or ``12s``."""
    if seconds < 0 or not math.isfinite(seconds):
        return "--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    if minutes:
        return f"{minutes}:{secs:02d}"
    return f"{secs}s"


def gpu_memory_gb() -> float:
    """Return peak allocated CUDA memory in GiB, or 0.0 off CUDA."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**30


@dataclass
class BatchProgress:
    """Per-batch YOLO-style progress bar for one epoch.

    The bar is **standalone**: it never wraps the DataLoader. Wrapping a
    ``DataLoader`` in ``tqdm`` perturbs the shuffling order even when the bar is
    disabled, which would make a run with progress enabled follow a different
    data order from one without. ``epoch()`` therefore returns the loader
    unchanged and the caller drives ``update()`` once per batch.
    """

    total_epochs: int
    total_batches: int
    enabled: bool = True
    _bar: object | None = field(default=None, init=False, repr=False)
    _running_loss: float = field(default=0.0, init=False, repr=False)
    _seen: int = field(default=0, init=False, repr=False)
    _colour: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._colour = supports_color() if self.enabled else False

    def header(self) -> None:
        """Print the column header once, before the first epoch."""
        if not self.enabled:
            return
        line = (
            f"{'Epoch':>9}{'GPU_mem':>9}{'loss':>10}{'lr_bb':>10}"
            f"{'lr_head':>10}{'batches':>12}"
        )
        print(_paint(line, BOLD, enabled=self._colour), flush=True)

    def epoch(self, iterable: Iterable, epoch: int) -> Iterator:
        """Open a bar for one epoch and return *iterable* untouched."""
        self._running_loss = 0.0
        self._seen = 0
        if self.enabled and _HAS_TQDM:
            self._bar = tqdm(
                total=self.total_batches,
                desc=f"{epoch}/{self.total_epochs}".rjust(9),
                bar_format=(
                    "{desc}{postfix} {bar} {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )
        return iter(iterable)

    def update(self, loss: float, lr_backbone: float, lr_head: float) -> None:
        """Advance the bar by one batch and refresh the running statistics."""
        self._seen += 1
        self._running_loss += (loss - self._running_loss) / self._seen
        if not self.enabled or self._bar is None:
            return
        postfix = (
            f"{gpu_memory_gb():>8.3g}G{self._running_loss:>10.4f}"
            f"{lr_backbone:>10.2e}{lr_head:>10.2e}"
        )
        self._bar.set_postfix_str(postfix, refresh=False)  # type: ignore[attr-defined]
        self._bar.update(1)  # type: ignore[attr-defined]

    def close(self) -> None:
        """Close the bar for this epoch."""
        if self._bar is not None:
            self._bar.close()  # type: ignore[attr-defined]
            self._bar = None

    @property
    def running_loss(self) -> float:
        """Return the running mean training loss for the current epoch."""
        return self._running_loss


class EpochTable:
    """Per-class validation table printed after each epoch."""

    def __init__(
        self,
        label_names: Sequence[str] = config.LABEL_NAMES,
        *,
        enabled: bool = True,
    ) -> None:
        self.label_names = list(label_names)
        self.enabled = enabled
        self._colour = supports_color() if enabled else False

    def render(self, metrics: dict[str, object], *, is_best: bool = False) -> str:
        """Return the formatted table for one validation pass."""
        width = max(12, max((len(name) for name in self.label_names), default=12) + 2)
        header = (
            f"{'Class':>{width}}{'Images':>9}{'P':>9}{'R':>9}{'F1':>9}"
        )
        lines = [_paint(header, BOLD, enabled=self._colour)]

        per_class = metrics.get("per_class", {}) or {}
        support_total = 0
        for name in self.label_names:
            entry = per_class.get(name, {})
            support = int(entry.get("support", 0))
            support_total += support
            lines.append(
                f"{name:>{width}}{support:>9}"
                f"{float(entry.get('precision', 0.0)):>9.4f}"
                f"{float(entry.get('recall', 0.0)):>9.4f}"
                f"{float(entry.get('f1', 0.0)):>9.4f}"
            )

        macro_precision = (
            sum(float(per_class.get(n, {}).get("precision", 0.0)) for n in self.label_names)
            / max(1, len(self.label_names))
        )
        macro_recall = (
            sum(float(per_class.get(n, {}).get("recall", 0.0)) for n in self.label_names)
            / max(1, len(self.label_names))
        )
        summary = (
            f"{'all':>{width}}{support_total:>9}"
            f"{macro_precision:>9.4f}{macro_recall:>9.4f}"
            f"{float(metrics.get('f1_macro', 0.0)):>9.4f}"
        )
        lines.append(_paint(summary, GREEN if is_best else CYAN, enabled=self._colour))

        tail = (
            f"  val_loss {float(metrics.get('loss', 0.0)):.4f}   "
            f"accuracy {float(metrics.get('accuracy', 0.0)):.4f}   "
            f"macro F1 {float(metrics.get('f1_macro', 0.0)):.4f}"
        )
        if is_best:
            tail += _paint("   <- best", GREEN, enabled=self._colour)
        lines.append(tail)
        return "\n".join(lines)

    def print(self, metrics: dict[str, object], *, is_best: bool = False) -> None:
        """Print the table if reporting is enabled."""
        if self.enabled:
            print(self.render(metrics, is_best=is_best), flush=True)


class ConfusionView:
    """Compact confusion matrix, printed at the end of a run."""

    def __init__(
        self,
        label_names: Sequence[str] = config.LABEL_NAMES,
        *,
        enabled: bool = True,
    ) -> None:
        self.label_names = list(label_names)
        self.enabled = enabled
        self._colour = supports_color() if enabled else False

    def render(self, matrix: Sequence[Sequence[int]]) -> str:
        """Return the confusion matrix with true rows and predicted columns."""
        short = [name[:7] for name in self.label_names]
        width = max(8, max((len(name) for name in short), default=8) + 1)
        header = " " * (width + 2) + "".join(f"{name:>{width}}" for name in short)
        lines = [
            _paint("Confusion matrix (rows = true, columns = predicted)", DIM, enabled=self._colour),
            _paint(header, BOLD, enabled=self._colour),
        ]
        for index, row in enumerate(matrix):
            label = short[index] if index < len(short) else str(index)
            cells = "".join(f"{int(value):>{width}}" for value in row)
            lines.append(f"{label:>{width + 2}}{cells}")
        return "\n".join(lines)

    def print(self, matrix: Sequence[Sequence[int]]) -> None:
        """Print the confusion matrix if reporting is enabled."""
        if self.enabled:
            print(self.render(matrix), flush=True)


class TrainingReporter:
    """Bundles the per-run displays and holds run-level context."""

    def __init__(
        self,
        run_id: str,
        *,
        total_epochs: int,
        total_batches: int,
        label_names: Sequence[str] = config.LABEL_NAMES,
        enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.enabled = enabled
        self.total_epochs = total_epochs
        self.bars = BatchProgress(total_epochs, total_batches, enabled=enabled)
        self.table = EpochTable(label_names, enabled=enabled)
        self.confusion = ConfusionView(label_names, enabled=enabled)
        self._colour = supports_color() if enabled else False
        self._started = time.perf_counter()

    def start(self, *, parameter_counts: dict | None = None, extra: str = "") -> None:
        """Print the run banner and the epoch-table header."""
        if not self.enabled:
            return
        rule = "=" * min(78, shutil.get_terminal_size((80, 20)).columns)
        print(_paint(rule, DIM, enabled=self._colour), flush=True)
        title = f"run {self.run_id}"
        if extra:
            title += f"   {extra}"
        print(_paint(title, BOLD, enabled=self._colour), flush=True)
        if parameter_counts:
            total = parameter_counts.get("total", {})
            print(
                f"  parameters  total {total.get('total', 0):,}   "
                f"trainable {total.get('trainable', 0):,}   "
                f"non-backbone {total.get('non_backbone', 0):,}",
                flush=True,
            )
        print(_paint(rule, DIM, enabled=self._colour), flush=True)
        self.bars.header()

    def epoch_iter(self, iterable: Iterable, epoch: int) -> Iterator:
        """Wrap the training loader for one epoch."""
        return self.bars.epoch(iterable, epoch)

    def batch(self, loss: float, lr_backbone: float, lr_head: float) -> None:
        """Report one optimisation micro-step."""
        self.bars.update(loss, lr_backbone, lr_head)

    def epoch_end(self, metrics: dict[str, object], *, is_best: bool) -> None:
        """Close the bar and print the per-class table."""
        self.bars.close()
        self.table.print(metrics, is_best=is_best)

    def finish(
        self,
        *,
        best_epoch: int,
        best_metrics: dict[str, float],
        confusion_matrix: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Print the closing summary for the run."""
        if not self.enabled:
            return
        if confusion_matrix is not None:
            self.confusion.print(confusion_matrix)
        elapsed = time.perf_counter() - self._started
        print(
            _paint(
                f"{self.total_epochs} epochs completed in {format_duration(elapsed)}   "
                f"best epoch {best_epoch}   "
                f"val macro F1 {best_metrics.get('f1_macro', float('nan')):.4f}   "
                f"val accuracy {best_metrics.get('accuracy', float('nan')):.4f}",
                BOLD,
                enabled=self._colour,
            ),
            flush=True,
        )


class SuiteBoard:
    """Live leaderboard across several runs, redrawn as each one completes."""

    COLUMNS = ("run", "status", "epochs", "trainable", "val F1", "val acc", "time")

    def __init__(self, run_ids: Sequence[str], *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._colour = supports_color() if enabled else False
        self.rows: dict[str, dict[str, object]] = {
            run_id: {
                "run": run_id,
                "status": "queued",
                "epochs": "-",
                "trainable": "-",
                "val F1": None,
                "val acc": None,
                "time": "-",
            }
            for run_id in run_ids
        }

    def mark_running(self, run_id: str) -> None:
        """Mark a run as in progress and redraw."""
        if run_id in self.rows:
            self.rows[run_id]["status"] = "running"
        self.render()

    def update(
        self,
        run_id: str,
        *,
        status: str,
        epochs: object = "-",
        trainable: object = "-",
        val_f1: float | None = None,
        val_accuracy: float | None = None,
        seconds: float | None = None,
    ) -> None:
        """Record a run's outcome and redraw."""
        self.rows[run_id] = {
            "run": run_id,
            "status": status,
            "epochs": epochs,
            "trainable": f"{trainable:,}" if isinstance(trainable, int) else trainable,
            "val F1": val_f1,
            "val acc": val_accuracy,
            "time": format_duration(seconds) if seconds is not None else "-",
        }
        self.render()

    def render(self) -> str:
        """Return the leaderboard, best macro F1 first."""
        def sort_key(row: dict[str, object]):
            score = row.get("val F1")
            return (0 if score is None else 1, score if score is not None else 0.0)

        ordered = sorted(self.rows.values(), key=sort_key, reverse=True)
        name_width = max(12, max((len(str(row["run"])) for row in ordered), default=12) + 2)
        header = (
            f"{'run':<{name_width}}{'status':>10}{'epochs':>8}{'trainable':>14}"
            f"{'val F1':>10}{'val acc':>10}{'time':>10}"
        )
        lines = [_paint(header, BOLD, enabled=self._colour), "-" * len(header)]
        for row in ordered:
            score = row.get("val F1")
            accuracy = row.get("val acc")
            status = str(row["status"])
            colour = (
                GREEN if status == "done" else YELLOW if status == "running" else DIM
            )
            lines.append(
                f"{str(row['run']):<{name_width}}"
                + _paint(f"{status:>10}", colour, enabled=self._colour)
                + f"{str(row['epochs']):>8}{str(row['trainable']):>14}"
                + (f"{score:>10.4f}" if isinstance(score, float) else f"{'-':>10}")
                + (f"{accuracy:>10.4f}" if isinstance(accuracy, float) else f"{'-':>10}")
                + f"{str(row['time']):>10}"
            )
        text = "\n".join(lines)
        if self.enabled:
            print("\n" + text + "\n", flush=True)
        return text
