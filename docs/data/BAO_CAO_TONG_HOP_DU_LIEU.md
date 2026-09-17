# Báo cáo tổng hợp dữ liệu dự án LDTF-BERT

## 1. Mục đích của báo cáo

Tài liệu này tổng hợp toàn bộ thông tin quan trọng về dữ liệu đang được dùng
trong dự án LDTF-BERT: nguồn gốc, ý nghĩa nhãn, quy mô, cách làm sạch, cách ghép
nguồn, kiểm soát trùng lặp, chia tập, kiểm tra chất lượng, cách model đọc dữ
liệu và các giới hạn còn lại.

Phiên bản dữ liệu chính thức hiện tại là **`multisource-v1`**. Đây là bộ dữ liệu
phân loại tin tức tiếng Anh gồm bốn chủ đề, được xây dựng từ AG News, BBC News
và HuffPost. AG News là dữ liệu nền; BBC News và HuffPost chỉ bổ sung cho tập
train. Validation và test được giữ nguyên từ AG News.

Trạng thái kiểm tra tự động: **PASS** và `ready_for_training=true`. Trạng thái
này xác nhận dữ liệu đạt các điều kiện kỹ thuật đã cấu hình, không đồng nghĩa
với việc mọi nhãn do nguồn cung cấp đều đúng tuyệt đối.

## 2. Tóm tắt nhanh

| Nội dung | Giá trị |
| --- | --- |
| Phiên bản | `multisource-v1` |
| Ngôn ngữ | Tiếng Anh |
| Bài toán | Phân loại tin tức một nhãn, bốn lớp |
| Dữ liệu nền | AG News |
| Dữ liệu bổ sung | BBC News và HuffPost |
| Train | 109.031 mẫu |
| Validation | 11.971 mẫu |
| Test | 7.600 mẫu |
| Tổng ba tập | 128.602 mẫu |
| Mẫu bổ sung vào train | 1.296 mẫu, 324 mẫu mỗi nhãn |
| Tokenizer | `bert-base-uncased` |
| Giới hạn đầu vào | 128 token |
| Seed lấy mẫu | 42 |
| Ngưỡng gần trùng | Jaccard >= 0,90 |
| Định dạng cuối | Apache Parquet |

## 3. Bài toán và hệ thống nhãn

Mỗi bài báo chỉ nhận một trong bốn nhãn sau:

| Mã | Nhãn | Nội dung |
| ---: | --- | --- |
| `0` | World | Tin thế giới, quan hệ quốc tế và sự kiện quốc tế |
| `1` | Sports | Tin thể thao |
| `2` | Business | Tin kinh doanh, doanh nghiệp và kinh tế |
| `3` | Sci/Tech | Tin khoa học và công nghệ |

Model nhận văn bản tiếng Anh và tạo bốn điểm số. Nhãn có điểm cao nhất được
chọn làm dự đoán. Dataset không được thiết kế cho tin tiếng Việt, phát hiện tin
giả, phân tích cảm xúc hoặc phân loại các chủ đề ngoài bốn lớp trên.

## 4. Nguồn dữ liệu

### 4.1. AG News

AG News là dữ liệu nền và cung cấp toàn bộ train gốc, validation và test. Dữ
liệu gồm tiêu đề, mô tả và nhãn từ 1 đến 4. Sau khi chuẩn hóa về quy ước của dự
án, nhãn được đưa về khoảng 0 đến 3.

| Nhãn AG News | Nhãn dự án |
| ---: | --- |
| `1` | World (`0`) |
| `2` | Sports (`1`) |
| `3` | Business (`2`) |
| `4` | Sci/Tech (`3`) |

Ba split AG News cố định được pipeline nhận vào có 107.735 mẫu train, 11.971
mẫu validation và 7.600 mẫu test.

### 4.2. BBC News

Nguồn BBC là bộ `BBC articles fulltext and category` trên Kaggle, gồm 2.225
bài. Pipeline chỉ nhận 1.422 bài thuộc ba chuyên mục phù hợp:

- Link dataset: [BBC articles fulltext and category trên Kaggle](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)

| Chuyên mục BBC | Nhãn dự án |
| --- | --- |
| `sport` | Sports (`1`) |
| `business` | Business (`2`) |
| `tech` | Sci/Tech (`3`) |

Các chuyên mục `politics` và `entertainment` bị loại vì không ánh xạ trực tiếp,
an toàn vào hệ bốn nhãn. File BBC không có cột tiêu đề riêng, vì vậy pipeline
dùng 16 từ đầu làm `title_clean`, phần còn lại làm `description_clean`. Cột
`text` vẫn giữ toàn bộ bài đã làm sạch và không lặp lại tiêu đề kỹ thuật.

### 4.3. HuffPost

Nguồn HuffPost là `News Category Dataset` trên Kaggle, gồm 209.527 bài. Pipeline
chỉ nhận 3.299 bài thuộc chuyên mục `WORLD NEWS` và ánh xạ thành World (`0`).
Các chuyên mục khác không được tự động đưa vào. Đặc biệt, pipeline không coi
mọi bài `POLITICS` là World vì hai khái niệm này không hoàn toàn tương đương.

- Link dataset: [News Category Dataset của HuffPost trên Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

Hai dataset trên chỉ được dùng để bổ sung dữ liệu cho **tập train AG News**.
Validation và test vẫn giữ nguyên từ AG News để việc so sánh kết quả trước và
sau khi bổ sung dữ liệu sử dụng cùng một thước đo.

### 4.4. Vai trò của từng nguồn trong bản cuối

| Nguồn | Vai trò trong train | Vai trò trong validation/test |
| --- | --- | --- |
| AG News | Dữ liệu nền của cả bốn nhãn | Toàn bộ validation và test |
| BBC News | Bổ sung Sports, Business, Sci/Tech | Không sử dụng |
| HuffPost | Bổ sung World | Không sử dụng |

Đường dẫn tải hai bộ bổ sung được ghi tại
`src/pipeline data processed/data/README.md`. Nội dung bài báo có thể vẫn thuộc
bản quyền của đơn vị xuất bản; cần kiểm tra điều khoản nguồn trước khi phân
phối lại.

## 5. Phân bố dữ liệu cuối

### 5.1. Theo split và nhãn

| Split | World | Sports | Business | Sci/Tech | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 27.298 | 27.289 | 27.238 | 27.206 | 109.031 |
| Validation | 2.997 | 2.996 | 2.991 | 2.987 | 11.971 |
| Test | 1.900 | 1.900 | 1.900 | 1.900 | 7.600 |

Các lớp gần cân bằng. Test cân bằng tuyệt đối; vì vậy Accuracy, Macro F1 và
Weighted F1 có thể được so sánh mà không bị một lớp lớn lấn át mạnh.

### 5.2. Thành phần train theo nguồn

| Nguồn | World | Sports | Business | Sci/Tech | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| AG News | 26.974 | 26.965 | 26.914 | 26.882 | 107.735 |
| BBC News | 0 | 324 | 324 | 324 | 972 |
| HuffPost | 324 | 0 | 0 | 0 | 324 |
| **Tổng** | **27.298** | **27.289** | **27.238** | **27.206** | **109.031** |

Phần bổ sung gồm đúng 1.296 bài, tương đương khoảng 1,19% train cuối. Việc giới
hạn phần bổ sung giúp tăng độ đa dạng nhưng tránh để phong cách BBC/HuffPost
lấn át AG News.

## 6. Ba file dữ liệu dùng cho thực nghiệm

| File | Vai trò | Quy tắc sử dụng |
| --- | --- | --- |
| `research_train.parquet` | Model học trọng số | Được phép đọc trong train |
| `research_validation.parquet` | Chọn epoch và cấu hình tốt nhất | Không cập nhật trọng số trực tiếp |
| `research_test.parquet` | Báo cáo kết quả cuối | Không dùng để chọn model hoặc tinh chỉnh |

Snapshot thuận tiện cho thực nghiệm nằm tại:

```text
data/processed_merg_AGnew_BBC_Huffpost/
```

Đầu ra chuẩn trực tiếp của pipeline nằm tại:

```text
src/pipeline data processed/data/processed/
```

Hai vị trí đã được đối chiếu bằng SHA-256 ở lần build hiện tại. Khi một bên bị
thay đổi hoặc sao chép lại, cần kiểm tra hash trước khi sử dụng.

## 7. Tại sao dùng Parquet thay vì CSV?

Dữ liệu nguồn có thể ở CSV hoặc JSON vì đây là định dạng phổ biến để trao đổi.
Đầu ra cuối chuyển sang Parquet vì:

- Parquet lưu kiểu dữ liệu rõ ràng, đặc biệt giữ `label` là số nguyên;
- đọc một số cột cần thiết nhanh hơn, ví dụ chỉ đọc `text` và `label`;
- nén tốt hơn đối với dữ liệu lặp cấu trúc, giúp giảm dung lượng và thời gian
  tải từ ổ đĩa hoặc Google Drive;
- ít gặp lỗi dấu phẩy, dấu ngoặc kép và xuống dòng nằm bên trong bài báo;
- phù hợp với Pandas và pipeline huấn luyện theo batch.

CSV vẫn hữu ích cho các báo cáo nhỏ cần mở bằng Excel. Vì vậy project dùng
Parquet cho dataset lớn và CSV/JSON cho báo cáo, manifest và dữ liệu kiểm tra.
Chuyển sang Parquet không làm model học tốt hơn trực tiếp; nó làm việc lưu trữ
và đọc dữ liệu ổn định, nhanh và ít lỗi hơn.

## 8. Schema và ý nghĩa từng cột

Ba file Parquet cuối dùng cùng schema, mỗi hàng là một bài báo.

| Cột | Kiểu/nhóm | Ý nghĩa |
| --- | --- | --- |
| `row_id` | Chuỗi | ID duy nhất có tiền tố nguồn |
| `source_dataset` | Chuỗi | `ag_news`, `bbc_news` hoặc `huffpost` |
| `source_id` | Chuỗi | ID hoặc liên kết tại nguồn ban đầu |
| `source_split` | Chuỗi | `train`, `validation`, `test` hoặc `external` |
| `original_label` | Chuỗi | Nhãn trước khi ánh xạ |
| `source_category` | Chuỗi | Tên chuyên mục tại nguồn |
| `title_raw` | Chuỗi | Tiêu đề trước làm sạch |
| `description_raw` | Chuỗi | Mô tả/nội dung trước làm sạch |
| `text_raw` | Chuỗi | Văn bản đầy đủ trước làm sạch |
| `label` | `int64` | Nhãn train từ 0 đến 3 |
| `label_name` | Chuỗi | Tên của nhãn |
| `title_clean` | Chuỗi | Tiêu đề sau làm sạch |
| `description_clean` | Chuỗi | Mô tả/nội dung sau làm sạch |
| `text` | Chuỗi | Văn bản model hiện đọc |
| `text_hash` | Chuỗi | SHA-256 văn bản chuẩn hóa để kiểm tra trùng |

Model hiện dùng `TEXT_ENCODING="single"`, nên đầu vào chính là `text`. Hai cột
`title_clean` và `description_clean` được giữ để có thể thử chế độ sentence
pair trong tương lai mà không phải xử lý lại dữ liệu.

## 9. Quy trình xử lý dữ liệu

```text
AG News cố định + BBC + HuffPost
              ↓
chuẩn hóa schema và ánh xạ nhãn
              ↓
làm sạch văn bản theo cùng quy tắc
              ↓
loại trùng hoàn toàn và nhãn mâu thuẫn
              ↓
phát hiện gần trùng bằng MinHash/LSH + Jaccard
              ↓
so sánh dữ liệu mới với train/validation/test AG News
              ↓
lấy mẫu cân bằng bằng seed 42
              ↓
ghép vào train, giữ nguyên validation và test
              ↓
xuất Parquet, manifest, checksum và báo cáo PASS
```

### 9.1. Chuẩn hóa từng nguồn

Mỗi adapter đọc định dạng riêng rồi đưa về 15 cột chung. Bước này giúp các
nguồn khác cấu trúc vẫn đi qua cùng quy trình kiểm tra.

### 9.2. Làm sạch văn bản

Pipeline thực hiện lần lượt:

1. chuyển giá trị thiếu thành chuỗi rỗng;
2. chuẩn hóa Unicode theo NFKC;
3. sửa và giải mã numeric HTML entity;
4. bảo vệ mã chứng khoán dạng `<MSFT.O>`;
5. đổi thẻ xuống dòng HTML thành khoảng trắng và bỏ thẻ HTML còn lại;
6. sửa ký tự escape như `\$`, `\"`, `\'`;
7. thu gọn khoảng trắng;
8. tạo `text_hash` từ văn bản chuẩn hóa và chuyển về chữ thường.

Mục tiêu lý thuyết là giảm nhiễu trình bày mà vẫn giữ nội dung ngữ nghĩa. Việc
dùng một quy tắc chung cũng làm giảm khác biệt giả tạo giữa các nguồn.

### 9.3. Loại trùng hoàn toàn

`text_hash` là dấu vân tay của văn bản đã chuẩn hóa:

- cùng hash và cùng nhãn: giữ một bản;
- cùng hash nhưng khác nhãn: loại toàn bộ nhóm vì nhãn mâu thuẫn;
- dữ liệu bổ sung trùng AG News: ưu tiên giữ AG News nền.

Kết quả đã loại 72 hàng trùng hoàn toàn cùng nhãn. Không phát hiện nhóm trùng
hoàn toàn nhưng khác nhãn trong dữ liệu bổ sung được chấp nhận.

### 9.4. Phát hiện gần trùng

Hai bài có thể gần giống nhau nhưng khác một số ký tự nên hash không giống.
Pipeline tách văn bản thành character 3-gram, dùng 64 phép hoán vị MinHash và
LSH để tìm ứng viên nhanh, sau đó xác nhận bằng Jaccard thực tế. Cặp có độ giống
từ 0,90 trở lên được xem là gần trùng.

Kết quả đã loại 40 hàng gần trùng cùng nhãn. Dữ liệu bổ sung không tạo overlap
gần trùng mới với validation hoặc test tại ngưỡng 0,90.

### 9.5. Lấy mẫu cân bằng

Sau lọc chất lượng, số ứng viên còn lại là:

| Nhãn | Ứng viên |
| --- | ---: |
| World | 3.296 |
| Sports | 489 |
| Business | 500 |
| Sci/Tech | 324 |

Cấu hình cho phép tối đa 400 bài mới mỗi nhãn, nhưng Sci/Tech chỉ còn 324. Để
không thiên lệch phần bổ sung, pipeline chọn 324 bài cho mỗi nhãn. Seed 42 bảo
đảm chạy lại sẽ chọn cùng danh sách nếu đầu vào và code không đổi.

### 9.6. Giữ nguyên validation và test

Train mới bằng AG News train cộng 1.296 bài bổ sung. Validation và test được sao
chép theo đúng danh sách ID cũ. Fingerprint trước và sau pipeline phải giống
nhau thì kiểm tra mới PASS.

Đây là cơ chế chống **data leakage**: model không được nhìn thấy nội dung tập
đánh giá trong lúc học. Nếu nội dung test lọt vào train, điểm test có thể cao
giả tạo và không còn phản ánh khả năng dự đoán dữ liệu chưa thấy.

## 10. Kết quả kiểm tra chất lượng

### 10.1. Làm sạch

Pipeline kiểm tra 132.027 hàng trước lấy mẫu cuối:

| Kiểm tra | Lỗi còn lại |
| --- | ---: |
| Văn bản rỗng | 0 |
| Thẻ HTML | 0 |
| Numeric HTML entity | 0 |
| Ký tự backslash lỗi | 0 |

Không có hàng nào bị loại vì rỗng sau làm sạch.

### 10.2. Trùng lặp và giao nhau giữa các split

| Kiểm tra | Kết quả |
| --- | --- |
| Trùng hoàn toàn trong train cuối | 0 |
| Trùng hoàn toàn cùng nhãn đã loại | 72 hàng |
| Gần trùng cùng nhãn đã loại | 40 hàng |
| Overlap mới với validation/test | 0 |
| Fingerprint validation được giữ | Có |
| Fingerprint test được giữ | Có |

Pipeline chỉ cam kết phần BBC/HuffPost mới không gây rò rỉ. Những hạn chế hoặc
gần trùng vốn có bên trong AG News nền được giữ nguyên để không thay đổi bộ đo
đánh giá lịch sử.

### 10.3. Độ dài token

| Thống kê | Giá trị |
| --- | ---: |
| Số hàng | 128.602 |
| Trung bình | 53,502 token |
| Trung vị | 49 token |
| P95 | 74 token |
| P99 | 128 token |
| Lớn nhất | 3.424 token |
| Vượt 128 token | 1% |

Ngưỡng chấp nhận là 5%, nên kiểm tra đạt PASS. Khi văn bản dài hơn 128 token,
tokenizer cắt phần cuối. Các bài rất dài chủ yếu đến từ BBC, do đó model có thể
bỏ lỡ thông tin quan trọng ở cuối bài.

## 11. Test, validation và nguyên tắc đánh giá

- Train dùng để cập nhật trọng số model.
- Validation dùng để chọn cấu hình và checkpoint tốt nhất.
- Test chỉ dùng sau khi mọi quyết định đã chốt.

Code huấn luyện thông thường chỉ tạo DataLoader cho train và validation. Tập
test được mở qua cơ chế guard riêng và mỗi lượt truy cập chính thức được ghi
log. Không nên đọc lỗi test rồi sửa model và báo lại điểm trên chính tập test
đó như một kết quả độc lập, vì khi ấy test đã gián tiếp trở thành validation.

Trong đợt thực nghiệm hiện tại, sáu cấu hình được so sánh trên validation bằng
cùng seed 42. B2 được chọn rồi mới đánh giá trên test AG News 7.600 mẫu. Kết
quả test này đánh giá khả năng tổng quát trên AG News, không đánh giá riêng khả
năng tổng quát sang BBC hoặc HuffPost.

## 12. Tính tái lập và checksum

Các tham số chính:

| Tham số | Giá trị |
| --- | --- |
| Pipeline | `multisource-v1` |
| Seed | 42 |
| Tokenizer | `bert-base-uncased` |
| `max_length` | 128 |
| Character n-gram | 3 |
| MinHash permutations | 64 |
| Ngưỡng Jaccard | 0,90 |

SHA-256 đầu ra chính thức:

| Split | SHA-256 |
| --- | --- |
| Train | `de4bddafcc5449b16a1c63b4dd89488c2ef15906194984b51812070e540d0601` |
| Validation | `78d96c6a809123ce4b0fed35d153939f3fc068dd0dcf139221faec3caec63fca` |
| Test | `ee6bad1e03b7cb2a71f1df88e91249d16c77b403339be867972b309217e0d797` |

Checksum giống nhau chứng minh nội dung file giống nhau từng byte. Nó không tự
chứng minh nhãn đúng về mặt ngữ nghĩa.

Các manifest quan trọng nằm tại:

```text
src/pipeline data processed/manifests/multisource/
```

Thư mục này lưu hash đầu vào/đầu ra, bản sao cấu hình, môi trường chạy, danh
sách ID từng split và các hàng bị loại cùng lý do.

## 13. Cách pipeline và model kết nối với nhau

```text
run_pipeline.py
    → tạo research_train/validation/test.parquet
    → src/config.py xác định đường dẫn
    → src/dataset.py chỉ đọc các cột cần thiết
    → tokenizer tạo input_ids và attention_mask
    → model BERT/LDTF-BERT nhận batch
    → train.py tối ưu trên train và đo validation
```

Với source hiện tại, `LDTF_DATA_DIR` phải trỏ đến thư mục cha của `processed`.
Đầu ra chuẩn của pipeline đã đúng cấu trúc này:

```powershell
$env:LDTF_DATA_DIR = (Resolve-Path -LiteralPath 'src/pipeline data processed/data').Path
```

Notebook nhóm dùng bản sao tại `LDTF_4_MEMBERS/data/processed/`, chỉ chứa train
và validation để tránh các thành viên train vô tình truy cập test. Test được
đóng gói riêng cho bước final evaluation.

## 14. Cách chạy lại pipeline

Từ thư mục gốc `LDTF_bert` trên Windows:

```powershell
& '.venv/Scripts/python.exe' 'src/pipeline data processed/scripts/run_pipeline.py'
```

Cấu hình thực tế:

```text
src/pipeline data processed/configs/pipeline_config.json
```

Kiểm thử pipeline:

```powershell
& '.venv/Scripts/python.exe' -m pytest `
  'src/pipeline data processed/Test/test_multisource_pipeline.py' -q
```

Không chỉnh trực tiếp dữ liệu trong `data/raw`. Nếu muốn thay quy tắc, chỉnh
code hoặc config rồi tạo lại standardized, merged, processed, reports và
manifests.

## 15. Cấu trúc thư mục dữ liệu

```text
src/pipeline data processed/
├── configs/                 # cấu hình pipeline
├── data/
│   ├── raw/                 # dữ liệu nguồn, không sửa trực tiếp
│   ├── standardized/        # từng nguồn sau chuẩn hóa schema
│   ├── merged/              # dữ liệu gộp trước/sau loại trùng
│   ├── processed/           # ba Parquet cuối
│   └── cache/huggingface/   # tokenizer cục bộ
├── manifests/multisource/   # ID, checksum, môi trường, hàng bị loại
├── reports/multisource/     # báo cáo chất lượng và phân bố
├── scripts/                 # code pipeline và adapters
└── Test/                    # kiểm thử pipeline
```

Các thư mục `raw`, `standardized` và `merged` phục vụ truy vết và tái tạo.
Model chỉ cần các file trong `processed`.

## 16. Giới hạn và rủi ro còn lại

1. **Source bias:** World bổ sung từ HuffPost, còn ba lớp kia bổ sung từ BBC.
   Model có thể học văn phong của nguồn cùng với chủ đề.
2. **Domain shift:** ba nguồn khác thời gian, độ dài và phong cách biên tập.
3. **Nhãn nguồn không hoàn hảo:** một số bài Business và Sci/Tech có thể giao
   thoa hoặc được biên tập gắn nhãn chưa nhất quán.
4. **Cắt văn bản:** 1% mẫu vượt 128 token; phần cuối không được model đọc.
5. **Gần trùng phụ thuộc ngưỡng:** Jaccard 0,90 không bắt được mọi trường hợp
   diễn đạt lại cùng nội dung.
6. **AG News nền được giữ nguyên:** pipeline không sửa các vấn đề lịch sử bên
   trong AG News để bảo toàn khả năng so sánh.
7. **Test chỉ từ AG News:** điểm test không trực tiếp chứng minh model hoạt
   động tốt trên BBC, HuffPost hoặc tin tức hiện đại ngoài domain này.
8. **Giấy phép và bản quyền:** cần kiểm tra điều khoản từng nguồn trước khi
   phát hành hoặc sử dụng thương mại.

## 17. Việc cần làm trước khi công bố dataset

Báo cáo tự động yêu cầu `manual_label_review_required_before_release=true`.
File `reports/multisource/manual_review_samples.csv` chứa 100 mẫu để duyệt bằng
mắt, gồm 25 mẫu cho mỗi cặp nguồn-nhãn. Người duyệt nên kiểm tra:

- nội dung có đúng nhãn không;
- Business và Sci/Tech có bị nhầm không;
- `WORLD NEWS` có thật sự là tin quốc tế không;
- phần đầu bài BBC có đủ thông tin khi giới hạn 128 token không.

Sau khi duyệt cần ghi người duyệt, ngày, số lỗi, tiêu chí và quyết định chấp
nhận trong một biên bản riêng. Không sửa trực tiếp file mẫu do pipeline tạo.

## 18. Kết luận

`multisource-v1` là bộ dữ liệu AG News được bổ sung có kiểm soát bằng 1.296 bài
từ BBC News và HuffPost. Tập train gần cân bằng, validation/test được bảo toàn,
không có overlap mới do dữ liệu bổ sung và tỷ lệ vượt giới hạn 128 token nằm
dưới ngưỡng cấu hình. Dữ liệu đủ điều kiện kỹ thuật để huấn luyện và so sánh
các cấu hình trong dự án.

Khi diễn giải kết quả, cần nói rõ model được train trên dữ liệu đa nguồn nhưng
được validation và test trên AG News. Trạng thái PASS là bằng chứng về tính hợp
lệ kỹ thuật và khả năng tái lập, không thay thế việc kiểm tra nhãn thủ công,
đánh giá domain shift hoặc xem xét giấy phép dữ liệu.

## 19. Tài liệu và báo cáo gốc dùng để đối chiếu

- `docs/data/DATASET_CARD.md`: nguồn, mục đích và giới hạn;
- `docs/data/DATA_DICTIONARY.md`: schema và ý nghĩa từng cột;
- `docs/data/PROCESSING_PIPELINE.md`: trình tự xử lý;
- `docs/data/DATA_QUALITY.md`: kết quả kiểm tra chất lượng;
- `docs/data/REPRODUCIBILITY.md`: checksum và cách tái lập;
- `src/pipeline data processed/reports/multisource/final_data_report.json`:
  trạng thái PASS chính thức;
- `src/pipeline data processed/manifests/multisource/dataset_versions.json`:
  hash đầu vào, cấu hình và đầu ra.
