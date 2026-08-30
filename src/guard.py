"""Seal around the official AG News test split.

The official test set must never influence architecture choice, hyper-parameter
choice, epoch choice, seed choice, or checkpoint selection. Enforcement:

1. :func:`src.dataset.build_dataloaders` never returns a test loader.
2. The test parquet is reachable through exactly one function,
   :func:`official_test_loader`, which requires an explicit environment unlock.
3. Access is refused while a training loop is active in this process.
4. Every access is appended to ``reports/official_test_access.jsonl``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import traceback
from pathlib import Path
from typing import Any

from . import config
from .utils import append_jsonl

UNLOCK_ENV_VAR = "LDTF_ALLOW_OFFICIAL_TEST"
UNLOCK_TOKEN = "I_AM_REPORTING_FINAL_RESULTS"

_TRAINING_ACTIVE = False


class OfficialTestAccessError(RuntimeError):
    """Raised when the official test split is requested illegitimately."""


def begin_training() -> None:
    """Mark a training loop as active; blocks official-test access."""
    global _TRAINING_ACTIVE
    _TRAINING_ACTIVE = True


def end_training() -> None:
    """Mark the training loop as finished."""
    global _TRAINING_ACTIVE
    _TRAINING_ACTIVE = False


def training_active() -> bool:
    """Return whether a training loop is currently running in this process."""
    return _TRAINING_ACTIVE


def is_unlocked() -> bool:
    """Return whether the process environment carries the unlock token."""
    return os.environ.get(UNLOCK_ENV_VAR) == UNLOCK_TOKEN


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_official_test(path: str | Path) -> None:
    """Raise if *path* is the official test split."""
    if Path(path).resolve() == config.PROCESSED_TEST.resolve():
        raise OfficialTestAccessError(
            "The official test split was requested through an unguarded loader. "
            "Use src.guard.official_test_loader() from experiments/final_eval.py."
        )


def _ledger_length() -> int:
    """Return the number of recorded official-test accesses."""
    ledger = config.TEST_ACCESS_LEDGER
    if not ledger.exists():
        return 0
    with ledger.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def record_access(*, reason: str, run_id: str, extra: dict[str, Any] | None = None) -> int:
    """Append an official-test access record and return its 1-based index."""
    index = _ledger_length() + 1
    record: dict[str, Any] = {
        "access_index": index,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_id": run_id,
        "reason": reason,
        "test_sha256": sha256_file(config.PROCESSED_TEST),
        "caller": "".join(traceback.format_stack()[-4:-1]),
    }
    if extra:
        record.update(extra)
    append_jsonl(record, config.TEST_ACCESS_LEDGER)
    return index


def official_test_loader(
    tokenizer,
    *,
    reason: str,
    run_id: str,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
):
    """Return the official-test DataLoader, or raise.

    Requires ``LDTF_ALLOW_OFFICIAL_TEST=I_AM_REPORTING_FINAL_RESULTS`` in the
    environment and no active training loop. Every successful call is recorded
    in the append-only ledger.
    """
    from .dataset import build_eval_dataloader, load_split  # local import: avoid cycle

    if not is_unlocked():
        raise OfficialTestAccessError(
            "The official test split is sealed. Export "
            f"{UNLOCK_ENV_VAR}={UNLOCK_TOKEN} only when producing final "
            "reported numbers, and never during model selection."
        )
    if _TRAINING_ACTIVE:
        raise OfficialTestAccessError(
            "Official test access is forbidden while a training loop is active."
        )
    if not reason or not run_id:
        raise OfficialTestAccessError("Both 'reason' and 'run_id' must be provided.")

    extra: dict[str, Any] = {}
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise OfficialTestAccessError(
                f"Checkpoint {checkpoint} does not exist; evaluate a frozen checkpoint."
            )
        extra["checkpoint"] = str(checkpoint)
        extra["checkpoint_sha256"] = sha256_file(checkpoint)

    record_access(reason=reason, run_id=run_id, extra=extra)
    frame = load_split(config.PROCESSED_TEST, _guard_ok=True)
    return build_eval_dataloader(frame, tokenizer, batch_size=batch_size, split_name="test")
