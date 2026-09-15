# Quy trình xử lý dữ liệu

## Luồng tổng quát

```text
AG News cố định + BBC + HuffPost
              ↓
       Chuẩn hóa từng nguồn
              ↓
       Làm sạch văn bản chung
              ↓
   Loại trùng trong dữ liệu bổ sung
              ↓
So sánh với AG News train/validation/test
              ↓
      Lấy mẫu cân bằng bốn nhãn
              ↓
   Ghép vào train, giữ validation/test
              ↓
 Xuất Parquet, manifest và báo cáo PASS
```

## 1. Đọc dữ liệu theo nguồn

Ba adapter nằm trong `scripts/adapters`:

- `ag_news.py` đọc ba split Parquet cố định đang dùng trong project;
- `bbc_news.py` chỉ nhận `sport`, `business`, `tech`;
- `huffpost.py` chỉ nhận `WORLD NEWS`.

Mỗi adapter đưa dữ liệu về schema chung trước khi các nguồn được gộp.

## 2. Làm sạch văn bản

Pipeline thực hiện theo thứ tự:

1. chuyển giá trị thiếu thành chuỗi rỗng;
2. chuẩn hóa Unicode NFKC;
3. sửa và giải mã numeric HTML entity;
4. bảo vệ mã chứng khoán dạng `<MSFT.O>`;
5. đổi thẻ xuống dòng HTML thành khoảng trắng và bỏ các thẻ HTML còn lại;
6. sửa ký tự escape như `\$`, `\"`, `\'`;
7. thu gọn khoảng trắng;
8. tạo `text_hash` từ văn bản đã chuẩn hóa và chuyển về chữ thường.

Hàng có `text` rỗng sau bước này bị loại và được ghi vào manifest.

## 3. Loại bản ghi trùng hoàn toàn

Trong dữ liệu bổ sung:

- cùng văn bản, cùng nhãn: giữ một bản;
- cùng văn bản, khác nhãn: loại toàn bộ nhóm để tránh nhãn mâu thuẫn.

Ứng viên trùng hoàn toàn với AG News train, validation hoặc test cũng bị loại.
Pipeline luôn ưu tiên giữ dữ liệu AG News nền.

## 4. Phát hiện gần trùng

Pipeline tạo character 3-gram, tính MinHash với 64 phép hoán vị và dùng LSH để
tìm ứng viên. Một cặp chỉ được xem là gần trùng khi độ giống Jaccard thực tế
đạt ít nhất `0.90`.

Kiểm tra được thực hiện:

- giữa các bài bổ sung với nhau;
- giữa bài bổ sung và AG News train;
- giữa bài bổ sung và validation/test.

Nhờ đó dữ liệu mới không đưa nội dung giống tập đánh giá vào train.

## 5. Lấy mẫu cân bằng

Cấu hình yêu cầu tối đa 400 bài mới cho mỗi nhãn. Sau khi lọc, số ứng viên còn
lại là:

| Nhãn | Ứng viên còn lại |
| --- | ---: |
| World | 3.296 |
| Sports | 489 |
| Business | 500 |
| Sci/Tech | 324 |

Sci/Tech có ít ứng viên nhất, vì vậy pipeline chọn 324 bài cho mỗi nhãn. Việc
lấy mẫu dùng seed 42 để chạy lại vẫn chọn cùng danh sách hàng.

## 6. Tạo các split cuối

- Train mới = AG News train cố định + 1.296 bài bổ sung.
- Validation được sao chép theo đúng danh sách AG News validation cũ.
- Test được sao chép theo đúng danh sách AG News test cũ.

Fingerprint của validation và test được tính trước và sau khi xử lý. Hai giá
trị phải giống nhau thì báo cáo mới PASS.

## 7. Kiểm tra tokenizer

Pipeline dùng tokenizer `bert-base-uncased`, `max_length=128`. Nó đo phân bố độ
dài không truncation, sau đó tạo một batch có truncation để kiểm tra shape,
attention mask và miền nhãn.

## 8. Chạy pipeline

Từ thư mục `LDTF_bert` trên Windows:

```powershell
& '.venv/Scripts/python.exe' 'src/pipeline data processed/scripts/run_pipeline.py'
```

Cấu hình nằm tại:

```text
src/pipeline data processed/configs/pipeline_config.json
```

Đầu ra chuẩn nằm trong `src/pipeline data processed/data/processed`. Pipeline
ghi báo cáo vào `reports/multisource` và manifest vào `manifests/multisource`.
