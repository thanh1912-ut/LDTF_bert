# Hướng dẫn huấn luyện LDTF-BERT với dữ liệu mới

Tài liệu này hướng dẫn sử dụng bộ dữ liệu đã ghép từ **AG News, BBC News và
HuffPost** để huấn luyện các mô hình trong dự án LDTF-BERT.

Thư mục dữ liệu đang sử dụng:

```text
data/processed_merg_AGnew_BBC_Huffpost/
```

> Bộ dữ liệu này đã được xử lý hoàn chỉnh. Không cần chạy lại pipeline xử lý dữ
> liệu trước khi train, trừ khi muốn tạo lại dataset từ dữ liệu gốc.

## 1. Bộ dữ liệu mới gồm những gì?

Dataset vẫn thực hiện bài toán phân loại tin tức tiếng Anh thành bốn nhãn:

| Mã nhãn | Tên nhãn | Nội dung |
| ---: | --- | --- |
| `0` | World | Tin thế giới và quan hệ quốc tế |
| `1` | Sports | Tin thể thao |
| `2` | Business | Tin kinh doanh và kinh tế |
| `3` | Sci/Tech | Tin khoa học và công nghệ |

Ba file dùng trong quá trình thực nghiệm:

| File | Số dòng | Vai trò |
| --- | ---: | --- |
| `research_train.parquet` | 109.031 | Dùng để mô hình học |
| `research_validation.parquet` | 11.971 | Chọn mô hình và epoch tốt nhất |
| `research_test.parquet` | 7.600 | Chỉ đánh giá một lần ở cuối |

Phân bố nhãn thực tế:

| Tập | World | Sports | Business | Sci/Tech |
| --- | ---: | ---: | ---: | ---: |
| Train | 27.298 | 27.289 | 27.238 | 27.206 |
| Validation | 2.997 | 2.996 | 2.991 | 2.987 |
| Test | 1.900 | 1.900 | 1.900 | 1.900 |

Tập train mới có thêm 1.296 bài:

- 324 bài World từ HuffPost;
- 324 bài Sports từ BBC News;
- 324 bài Business từ BBC News;
- 324 bài Sci/Tech từ BBC News.

Validation và test vẫn là dữ liệu AG News cũ, nhờ đó kết quả trước và sau khi
bổ sung dữ liệu có thể được so sánh trên cùng một thước đo.

## 2. Source code sử dụng dữ liệu như thế nào?

Luồng chạy cơ bản:

```text
experiments/run_experiment.py
        ↓
src/config.py xác định đường dẫn dữ liệu
        ↓
src/dataset.py đọc train và validation từ Parquet
        ↓
Tokenizer chuyển văn bản thành token của BERT
        ↓
src/registry.py tạo mô hình theo mã thí nghiệm
        ↓
src/train.py huấn luyện, đánh giá validation và lưu checkpoint
        ↓
outputs/<tên_lần_chạy>/
```

Ý nghĩa những file source code chính:

| Source code | Nhiệm vụ |
| --- | --- |
| `src/config.py` | Khai báo đường dẫn, tên cột, số nhãn và tham số mặc định |
| `src/dataset.py` | Đọc Parquet, lấy cột `text` và `label`, tokenize và tạo batch |
| `src/models/` | Chứa kiến trúc BERT, router và mô hình LDTF-BERT |
| `src/registry.py` | Ánh xạ mã `A0`, `B2`,... sang đúng cấu hình mô hình |
| `src/train.py` | Huấn luyện, tính validation và lưu checkpoint |
| `experiments/run_experiment.py` | Lệnh chạy một thí nghiệm |
| `experiments/run_suite.py` | Chạy liên tiếp nhiều thí nghiệm |
| `experiments/final_eval.py` | Đánh giá checkpoint đã chọn trên test chính thức |

Trong lúc train, chương trình chỉ đọc:

```text
research_train.parquet
research_validation.parquet
```

`research_test.parquet` được khóa và không tham gia học hoặc lựa chọn mô hình.

## 3. Chuẩn bị môi trường

Mở PowerShell và đi tới thư mục dự án:

```powershell
Set-Location "E:\Các môn học năm 2\Học sâu ver 2\LDTF_bert"
```

Môi trường `.venv` hiện tại của dự án được quản lý bằng `uv`. Nếu chưa cài đủ
thư viện, chạy:

```powershell
& "..\.tools\uv\bin\uv.exe" pip install `
    --python ".\.venv\Scripts\python.exe" `
    -r requirements.txt
```

Trên máy khác có sẵn `pip`, có thể dùng:

```powershell
python -m pip install -r requirements.txt
```

## 4. Trỏ source code tới dataset mới

`src/config.py` yêu cầu biến `LDTF_DATA_DIR` trỏ tới một thư mục cha có thư
mục con tên chính xác là `processed`.

Trong khi đó dataset hiện nằm tại:

```text
data/processed_merg_AGnew_BBC_Huffpost/
```

Vì vậy, không đặt `LDTF_DATA_DIR` trực tiếp bằng đường dẫn trên. Nếu làm vậy,
source sẽ tìm nhầm đường dẫn:

```text
data/processed_merg_AGnew_BBC_Huffpost/processed/
```

### Thiết lập một lần

Các lệnh dưới đây tạo một directory junction. Đây là một đường dẫn trung gian
trỏ tới dataset gốc, không sao chép và không ghi đè ba file Parquet:

```powershell
$runtimeDir = Join-Path (Get-Location) "data\multisource_runtime"
$targetDir = (Resolve-Path "data\processed_merg_AGnew_BBC_Huffpost").Path

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

if (-not (Test-Path (Join-Path $runtimeDir "processed"))) {
    New-Item -ItemType Junction `
        -Path (Join-Path $runtimeDir "processed") `
        -Target $targetDir | Out-Null
}
```

Sau bước này, đường dẫn mà source code nhìn thấy là:

```text
data/multisource_runtime/processed/
```

nhưng dữ liệu thực tế vẫn được lưu tại
`data/processed_merg_AGnew_BBC_Huffpost/`.

### Thiết lập cho mỗi cửa sổ PowerShell mới

Biến môi trường sau chỉ có hiệu lực trong cửa sổ PowerShell hiện tại:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path "data\multisource_runtime").Path
```

Nếu đóng PowerShell rồi mở lại để train, cần chạy lại lệnh này.

## 5. Kiểm tra source đã nhận đúng dữ liệu

Chạy lệnh sau trước khi train:

```powershell
.\.venv\Scripts\python.exe -c "from src import config; print('Train:', config.PROCESSED_TRAIN); print('Validation:', config.PROCESSED_VAL); print('Test:', config.PROCESSED_TEST)"
```

Ba đường dẫn được in ra phải bắt đầu bằng:

```text
...\data\multisource_runtime\processed\
```

Kiểm tra số dòng train và validation:

```powershell
.\.venv\Scripts\python.exe -c "from src import config; from src.dataset import load_split; print('Train:', len(load_split(config.PROCESSED_TRAIN))); print('Validation:', len(load_split(config.PROCESSED_VAL)))"
```

Kết quả đúng:

```text
Train: 109031
Validation: 11971
```

Không cần mở tập test trong bước kiểm tra này.

## 6. Chạy thử trước khi train đầy đủ

Lần chạy dưới đây chỉ lấy 2.000 dòng train và chạy một epoch. Mục đích là kiểm
tra đường dẫn dữ liệu, tokenizer, mô hình và GPU có hoạt động hay không:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiment `
    --run B2_bert_finetuned_cls `
    --seed 42 `
    --epochs 1 `
    --limit-train-rows 2000 `
    --output-dir outputs\multisource_v1_debug_B2_seed42
```

Kết quả của lần chạy này chỉ dùng để kiểm tra kỹ thuật, không dùng làm kết quả
nghiên cứu chính thức.

## 7. Train đầy đủ trên một GPU

### Bước 1: Train mô hình BERT đối chứng

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiment `
    --run B2_bert_finetuned_cls `
    --seed 42 `
    --batch-size 32 `
    --eval-batch-size 64 `
    --output-dir outputs\multisource_v1_B2_seed42
```

`B2_bert_finetuned_cls` là BERT phân loại thông thường. Nên chạy mô hình này để
biết LDTF-BERT có thật sự tốt hơn một BERT tiêu chuẩn hay không.

### Bước 2: Train mô hình LDTF-BERT chính

```powershell
.\.venv\Scripts\python.exe -m experiments.run_experiment `
    --run A0 `
    --seed 42 `
    --batch-size 32 `
    --eval-batch-size 64 `
    --output-dir outputs\multisource_v1_A0_seed42
```

`A0` là cấu hình LDTF-BERT tham chiếu chính. Mặc định mô hình fine-tune trong ba
epoch và chọn checkpoint có `validation macro F1` tốt nhất.

Tên output có tiền tố `multisource_v1_` để không nhầm checkpoint của dataset
mới với checkpoint từng train bằng AG News cũ.

## 8. Train bằng hai GPU

Khi có hai GPU, dùng `torchrun`:

```powershell
.\.venv\Scripts\python.exe -m torch.distributed.run `
    --standalone `
    --nproc_per_node=2 `
    -m experiments.run_experiment `
    --run A0 `
    --seed 42 `
    --batch-size 32 `
    --eval-batch-size 64 `
    --num-workers 1 `
    --pad-to-multiple-of 8 `
    --output-dir outputs\multisource_v1_A0_seed42
```

`--batch-size 32` là batch tổng. Với hai GPU, mỗi GPU xử lý 16 mẫu trong một
bước. Do train có 109.031 dòng, sampler phân tán có thể lặp lại một dòng mỗi
epoch để hai GPU nhận số batch bằng nhau; thông tin này được ghi trong báo cáo
của lần chạy.

## 9. Tiếp tục khi quá trình train bị gián đoạn

Nếu thư mục output đã có `last.pt`, dùng đúng cấu hình và đúng dataset rồi thêm
`--resume`:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path "data\multisource_runtime").Path

.\.venv\Scripts\python.exe -m experiments.run_experiment `
    --run A0 `
    --seed 42 `
    --batch-size 32 `
    --eval-batch-size 64 `
    --output-dir outputs\multisource_v1_A0_seed42 `
    --resume
```

Không dùng `--force` khi muốn tiếp tục checkpoint. `--force` có nghĩa là bắt
đầu lại một lần train mới và đưa kết quả cũ vào `previous_runs/`.

## 10. Chạy nhiều seed để kết quả đáng tin cậy hơn

Một seed có thể cho kết quả tốt hoặc xấu do ngẫu nhiên. Bộ seed chuẩn của dự án
là `42`, `1337` và `2024`.

Ví dụ với A0, chạy lại lệnh train và thay đồng thời:

```text
--seed 42    → outputs/multisource_v1_A0_seed42
--seed 1337  → outputs/multisource_v1_A0_seed1337
--seed 2024  → outputs/multisource_v1_A0_seed2024
```

Sau đó báo cáo trung bình và độ dao động của validation/test thay vì chỉ chọn
seed có kết quả cao nhất.

Đây là quy trình đầy đủ khi có đủ thời gian tính toán. Đợt Colab hiện tại ưu
tiên hoàn thành sáu kiến trúc trong thời gian ngắn hơn, nên cả ba member dùng
chung seed 42 và mỗi người chạy hai model. Vì vậy báo cáo hiện tại dùng để so
sánh kiến trúc trên cùng seed, chưa đo được độ ổn định qua nhiều seed.

### Phân công nhóm bốn thành viên trên Google Colab

Dự án có một notebook train tại
`notebooks/colab_4_members_multisource.ipynb`, một notebook phân tích tại
`notebooks/colab_member_4_analysis.ipynb` và cấu hình chung tại
`configs/colab_4_members.json`:

| Thành viên | Vai trò |
| --- | --- |
| Member 1 | Train `B2, A0` với seed 42 |
| Member 2 | Train `A1, A3` với seed 42 |
| Member 3 | Train `A4, A11` với seed 42 |
| Member 4 | Kiểm tra đủ 6 model và tổng hợp validation |

Member 1-3 mở notebook train, chỉ thay `MEMBER_ID` thành 1, 2 hoặc 3 và bật GPU
T4. Member 4 mở notebook phân tích riêng và có thể dùng CPU. Cả hai notebook
đều không sử dụng tập test.

Mỗi thành viên sử dụng Google Drive riêng. Khi chạy notebook, code tự tạo
`MyDrive/LDTF_4_MEMBERS` cùng các thư mục `source`, `data`, `results`, `exports`,
`incoming` và `analysis`. Folder bàn giao đã chứa sẵn `LDTF_bert_source.zip`,
train và validation. Mỗi người upload nguyên folder `LDTF_4_MEMBERS` vào
`MyDrive`. Member 1-3 mở notebook train, sửa `MEMBER_ID`, bật GPU T4 và bấm
**Run all**. Member 4 mở `colab_member_4_analysis.ipynb` rồi bấm **Run all**.
Không cần clone hoặc pull source từ GitHub.

Cả bốn người phải bắt đầu từ cùng folder bàn giao để checksum source và dữ liệu
giống nhau. Notebook train sẽ dừng với đường dẫn file còn thiếu để tránh vô tình
dùng sai dữ liệu.

Sau khi train, mỗi trainer chạy bước bàn giao để tạo một file ZIP summary nhỏ
trong `exports/` rồi gửi cho Member 4. Member 4 tải ba file nhận được vào
`incoming/`; notebook phân tích sẽ tự nhập, kiểm tra protocol và tổng hợp đủ 6
kết quả. Checkpoint đầy đủ vẫn được giữ trên Drive cá nhân của từng trainer.

Sau mỗi epoch hoàn tất, pipeline cập nhật `last.pt`, `train_log.jsonl` và
`val_metrics.json` trực tiếp trên Drive. Khi phiên Colab bị ngắt, mở lại bằng
cùng `MEMBER_ID` và Run all; pipeline tự thêm `--resume`. Nếu phiên bị ngắt giữa
một epoch thì epoch đang dở chạy lại từ đầu, còn các epoch đã hoàn tất không bị
mất. `best.pt` và `last.pt` đều được giữ sau khi train xong.

## 11. Các file kết quả được tạo ra

Ví dụ trong `outputs/multisource_v1_A0_seed42/`:

| File | Ý nghĩa |
| --- | --- |
| `best.pt` | Checkpoint tốt nhất theo validation macro F1 |
| `last.pt` | Trạng thái epoch gần nhất, dùng cho `--resume` |
| `train_log.jsonl` | Loss và metric theo từng epoch |
| `val_metrics.json` | Chỉ số trên validation |
| `run_summary.json` | Tóm tắt cấu hình, thời gian, metric và dấu vân tay dữ liệu |

`run_summary.json` lưu SHA-256 của train và validation. Nhờ vậy có thể chứng
minh checkpoint đã được train bằng đúng phiên bản dữ liệu mới.

## 12. Đánh giá test chính thức

Chỉ thực hiện bước này sau khi đã:

1. train xong tất cả mô hình và seed;
2. chọn mô hình bằng validation;
3. không còn thay đổi kiến trúc hoặc tham số dựa trên kết quả test.

Đánh giá A0 đã chọn:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path "data\multisource_runtime").Path
$env:LDTF_ALLOW_OFFICIAL_TEST = "I_AM_REPORTING_FINAL_RESULTS"

.\.venv\Scripts\python.exe -m experiments.final_eval `
    --run multisource_v1_A0_seed42 `
    --reason "Danh gia cuoi cung tren bo du lieu multisource-v1"
```

Kết quả test được ghi vào:

```text
outputs/multisource_v1_A0_seed42/test_metrics.json
outputs/multisource_v1_A0_seed42/test_predictions.npz
reports/final_test_metrics.json
reports/official_test_access.jsonl
```

Mỗi lần mở test đều được ghi vào `official_test_access.jsonl`. Không dùng test
để quyết định model nào nên train lại.

## 13. Cách chạy trên Kaggle

Khi đưa dataset lên Kaggle, cấu trúc nên là:

```text
/kaggle/input/<ten-dataset>/processed/
  research_train.parquet
  research_validation.parquet
  research_test.parquet
```

Trong notebook, thiết lập:

```python
import os

os.environ["LDTF_DATA_DIR"] = "/kaggle/input/<ten-dataset>"
```

Sau đó chạy với hai GPU T4:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m experiments.run_experiment --run A0 --seed 42 \
    --batch-size 32 --eval-batch-size 64 --num-workers 1 \
    --pad-to-multiple-of 8 \
    --output-dir outputs/multisource_v1_A0_seed42
```

Không chạy `experiments.final_eval` bằng `torchrun`; đánh giá test chỉ chạy một
process Python thông thường.

## 14. Những điều cần nhớ

- Luôn thiết lập `LDTF_DATA_DIR` trước khi chạy train trong terminal mới.
- Xác nhận train có 109.031 dòng trước lần chạy đầy đủ.
- Đặt tên output chứa `multisource_v1` để phân biệt với dataset cũ.
- Chỉ dùng validation để chọn mô hình và epoch.
- Không mở test cho tới khi toàn bộ quyết định huấn luyện đã hoàn tất.
- Không so sánh hai mô hình nếu chúng dùng seed, batch size hoặc dữ liệu khác
  nhau mà không ghi rõ sự khác biệt.

## 15. Tài liệu liên quan

- `redmee.md`: tổng quan kiến trúc, source code và quy trình thí nghiệm.
- `docs/data/README.md`: thứ tự đọc toàn bộ tài liệu dữ liệu.
- `docs/data/DATASET_CARD.md`: nguồn, thành phần và giới hạn của dataset mới.
- `docs/data/PROCESSING_PIPELINE.md`: cách ba nguồn dữ liệu được xử lý và ghép.
- `docs/data/DATA_QUALITY.md`: các kiểm tra chất lượng đã thực hiện.
- `docs/data/REPRODUCIBILITY.md`: checksum và cách tái tạo dataset.
