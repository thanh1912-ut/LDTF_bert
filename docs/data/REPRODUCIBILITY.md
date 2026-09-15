# Tái lập bộ dữ liệu

## Phiên bản

- Phiên bản pipeline: `multisource-v1`.
- Seed lấy mẫu: `42`.
- Tokenizer: `bert-base-uncased`.
- Giới hạn token: `128`.
- Ngưỡng gần trùng: `0,90`.
- Số phép hoán vị MinHash: `64`.

Toàn bộ giá trị được lưu trong
`src/pipeline data processed/configs/pipeline_config.json`.

## Môi trường

Môi trường Python cục bộ nằm tại `LDTF_bert/.venv`. Có thể cài lại các thư viện
theo file requirements của project và pipeline.

Lệnh chạy từ thư mục `LDTF_bert`:

```powershell
& '.venv/Scripts/python.exe' 'src/pipeline data processed/scripts/run_pipeline.py'
```

Không sửa file trong `data/raw`. Mỗi lần chạy tạo lại standardized data, merged
data, ba split cuối, báo cáo và manifest từ đầu vào đã khai báo.

## Kiểm tra tính toàn vẹn

Các file sau phục vụ tái lập:

| File | Vai trò |
| --- | --- |
| `manifests/multisource/raw_checksums.txt` | SHA-256 của từng đầu vào |
| `manifests/multisource/dataset_versions.json` | Phiên bản, hash cấu hình và hash đầu ra |
| `manifests/multisource/pipeline_config.json` | Bản cấu hình được chụp tại lúc chạy |
| `manifests/multisource/environment.json` | Phiên bản Python, Pandas, NumPy và hệ điều hành |
| `manifests/multisource/train_ids.json` | Danh sách ID train cuối |
| `manifests/multisource/validation_ids.json` | Danh sách ID validation |
| `manifests/multisource/test_ids.json` | Danh sách ID test |
| `manifests/multisource/removed_rows.csv` | Hàng bị loại và lý do |

Hash đầu ra của lần build hiện tại:

| Split | SHA-256 |
| --- | --- |
| Train | `de4bddafcc5449b16a1c63b4dd89488c2ef15906194984b51812070e540d0601` |
| Validation | `78d96c6a809123ce4b0fed35d153939f3fc068dd0dcf139221faec3caec63fca` |
| Test | `ee6bad1e03b7cb2a71f1df88e91249d16c77b403339be867972b309217e0d797` |

## Vị trí dữ liệu

Đầu ra chính của pipeline:

```text
src/pipeline data processed/data/processed/
```

Một bản sao đã được đối chiếu hash nằm tại:

```text
data/processed_merg_AGnew_BBC_Huffpost/
```

Tại thời điểm kiểm tra ngày 15-09-2026, ba file trong hai vị trí giống nhau.
Nếu một bên được chạy hoặc sao chép lại, cần đối chiếu SHA-256 lần nữa.

## Cho model sử dụng đầu ra pipeline

Code model xem `LDTF_DATA_DIR` là thư mục cha của folder `processed`. Với đầu ra
chuẩn của pipeline, dùng:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path -LiteralPath 'src/pipeline data processed/data').Path
```

Sau đó chạy lệnh huấn luyện trong cùng cửa sổ PowerShell. Folder
`data/processed_merg_AGnew_BBC_Huffpost` hiện chứa trực tiếp ba file nên không
khớp quy ước `LDTF_DATA_DIR/processed`; nó là bản snapshot để lưu trữ và đối
chiếu, trừ khi cấu trúc được thêm một cấp `processed` hoặc code đường dẫn được
đổi.

## Kiểm thử

```powershell
& '.venv/Scripts/python.exe' -m pytest `
  'src/pipeline data processed/Test/test_multisource_pipeline.py' -q
```

Lần triển khai hiện tại có 5 kiểm thử và tất cả đều đạt. Đầu ra train cũng đã
được nạp thử bằng `AgNewsTextDataset` và `BatchedTokenizingCollator` của project.
