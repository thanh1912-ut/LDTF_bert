"""Build one self-contained Google Drive folder for the Colab team.

The generated folder contains only a notebook, a compact source archive, and
the processed train/validation files. It intentionally excludes .git, virtual
environments, caches, checkpoints, reports, and the sealed test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = ROOT_DIR.parent
DEFAULT_OUTPUT = WORKSPACE_DIR / "LDTF_COLAB_READY" / "LDTF_4_MEMBERS"
TRAIN_NOTEBOOK = ROOT_DIR / "notebooks" / "colab_4_members_multisource.ipynb"
ANALYSIS_NOTEBOOK = ROOT_DIR / "notebooks" / "colab_member_4_analysis.ipynb"
DATA_DIR = ROOT_DIR / "data" / "processed_merg_AGnew_BBC_Huffpost"
SOURCE_ARCHIVE_NAME = "LDTF_bert_source.zip"
ARCHIVE_ROOT = "LDTF_bert_source"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> list[Path]:
    files: list[Path] = []
    files.extend((ROOT_DIR / "src").glob("*.py"))
    files.extend((ROOT_DIR / "src" / "models").glob("*.py"))
    files.extend((ROOT_DIR / "experiments").glob("*.py"))
    files.extend(
        path
        for path in (ROOT_DIR / "scripts").glob("*.py")
        if not path.name.startswith("build_colab")
    )
    files.extend((ROOT_DIR / "configs").glob("*.json"))
    files.extend(
        path
        for name in ("requirements.txt", "pytest.ini", "README.md")
        if (path := ROOT_DIR / name).is_file()
    )
    return sorted(
        (path for path in files if "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(ROOT_DIR).as_posix(),
    )


def write_source_archive(destination: Path) -> dict[str, object]:
    files = source_files()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT_DIR).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return {
        "filename": destination.name,
        "sha256": sha256_file(destination),
        "files": len(files),
        "bytes": destination.stat().st_size,
    }


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise ValueError(f"Unexpected checksum for {source}: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != actual:
        shutil.copy2(source, destination)


def guide_text() -> str:
    return """# BẮT ĐẦU Ở ĐÂY

Folder này dùng chung về cấu trúc, nhưng mỗi thành viên upload một bản vào
Google Drive cá nhân của mình.

## Cách chạy cho Member 1-3

1. Upload nguyên folder `LDTF_4_MEMBERS` vào `MyDrive`.
2. Chờ Google Drive tải xong đủ bốn file chính.
3. Mở `colab_4_members_multisource.ipynb` bằng Google Colab.
4. Sửa `MEMBER_ID` thành 1, 2 hoặc 3 theo phân công.
5. Chọn `Runtime > Change runtime type > T4 GPU`.
6. Chọn `Runtime > Run all` và cấp quyền Google Drive khi Colab hỏi.

Không đổi seed, model, batch size hoặc epoch. Cả ba member dùng seed 42;
notebook tự lấy đúng hai model được phân công từ
`MEMBER_ID`, kiểm tra checksum, copy dữ liệu vào ổ Colab, resume job bị ngắt và
lưu log và checkpoint về Drive cá nhân sau từng epoch hoàn tất. Khi chạy lại,
notebook tự tiếp tục từ `last.pt`; `best.pt` luôn là checkpoint có validation
Macro F1 tốt nhất.

Sau khi train, lấy file trong `exports/` gửi cho Member 4:

- Member 1 (`B2`, `A0`): `member_1_seed42_summaries.zip`
- Member 2 (`A1`, `A3`): `member_2_seed42_summaries.zip`
- Member 3 (`A4`, `A11`): `member_3_seed42_summaries.zip`

Không gửi `best.pt` hoặc `last.pt` trong bước tổng hợp validation.

## Member 4

Member 4 mở notebook riêng `colab_member_4_analysis.ipynb`, tải ba ZIP summary
nhận được vào `incoming/`, rồi Run all để kiểm tra protocol, tổng hợp bảng,
xếp hạng 6 model trên seed 42 và tạo biểu đồ trong `analysis/`.

Tập test không nằm trong folder bàn giao và không được sử dụng ở giai đoạn này.
"""


def build(output: Path) -> dict[str, object]:
    config = json.loads(
        (ROOT_DIR / "configs" / "colab_4_members.json").read_text(encoding="utf-8")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    (output / "data" / "processed").mkdir(parents=True, exist_ok=True)

    for notebook in (TRAIN_NOTEBOOK, ANALYSIS_NOTEBOOK):
        if not notebook.is_file():
            raise FileNotFoundError(notebook)
        shutil.copy2(notebook, output / notebook.name)
    source_manifest = write_source_archive(output / "source" / SOURCE_ARCHIVE_NAME)

    dataset_cfg = config["dataset"]
    for filename in (dataset_cfg["train_file"], dataset_cfg["validation_file"]):
        copy_verified(
            DATA_DIR / filename,
            output / "data" / "processed" / filename,
            dataset_cfg["sha256"][filename],
        )

    (output / "BAT_DAU_O_DAY.md").write_text(guide_text(), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "folder_name_on_drive": "LDTF_4_MEMBERS",
        "notebooks": {
            "train_members_1_2_3": TRAIN_NOTEBOOK.name,
            "analysis_member_4": ANALYSIS_NOTEBOOK.name,
        },
        "source": source_manifest,
        "data": {
            filename: {
                "sha256": dataset_cfg["sha256"][filename],
                "bytes": (output / "data" / "processed" / filename).stat().st_size,
            }
            for filename in (dataset_cfg["train_file"], dataset_cfg["validation_file"])
        },
        "sealed_test_included": False,
    }
    (output / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ready-to-upload Colab folder.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args.output.resolve())
    total_bytes = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    print(f"[colab-ready] folder: {args.output.resolve()}")
    print(f"[colab-ready] source files: {manifest['source']['files']}")
    print(f"[colab-ready] total: {total_bytes / (1024 ** 2):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
