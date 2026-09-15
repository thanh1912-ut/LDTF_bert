# Thẻ mô tả bộ dữ liệu

## Tổng quan

Bộ dữ liệu `multisource-v1` dùng cho bài toán phân loại tin tức tiếng Anh thành
bốn chủ đề của AG News. AG News là dữ liệu nền; BBC News và HuffPost chỉ được
dùng để bổ sung cho tập huấn luyện.

| Nhãn | Mã | Ý nghĩa |
| --- | ---: | --- |
| World | 0 | Tin thế giới, quan hệ quốc tế và sự kiện quốc tế |
| Sports | 1 | Tin thể thao |
| Business | 2 | Tin kinh doanh, doanh nghiệp và kinh tế |
| Sci/Tech | 3 | Tin khoa học và công nghệ |

## Thành phần cuối

| Tập | Số mẫu | Vai trò |
| --- | ---: | --- |
| Train | 109.031 | Huấn luyện mô hình |
| Validation | 11.971 | Chọn cấu hình và epoch tốt nhất |
| Test | 7.600 | Đánh giá cuối cùng, không dùng để điều chỉnh mô hình |

Tập train gồm 107.735 mẫu AG News có sẵn và 1.296 mẫu bổ sung:

| Nguồn | Nhãn được dùng | Số mẫu được chọn |
| --- | --- | ---: |
| HuffPost | World | 324 |
| BBC News | Sports | 324 |
| BBC News | Business | 324 |
| BBC News | Sci/Tech | 324 |

Validation và test chỉ chứa AG News. Danh sách mẫu của hai tập này được giữ
nguyên để kết quả trước và sau khi bổ sung dữ liệu có thể so sánh trực tiếp.

## Nguồn dữ liệu

- AG News: bộ dữ liệu nền gồm tiêu đề, mô tả và một trong bốn nhãn mục tiêu.
- BBC News: bản `BBC articles fulltext and category` trên Kaggle, gồm 2.225 bài.
  Pipeline chỉ nhận các chuyên mục `sport`, `business` và `tech`.
- HuffPost: bản `News Category Dataset` trên Kaggle, gồm 209.527 bài. Pipeline
  chỉ nhận chuyên mục `WORLD NEWS`.

Đường dẫn nguồn được ghi trong
`src/pipeline data processed/data/README.md`. Checksum SHA-256 của từng đầu vào
được lưu tại `manifests/multisource/raw_checksums.txt`.

## Quy tắc ánh xạ nhãn

| Nguồn | Nhãn gốc | Nhãn chung |
| --- | --- | --- |
| AG News | 1 | World (0) |
| AG News | 2 | Sports (1) |
| AG News | 3 | Business (2) |
| AG News | 4 | Sci/Tech (3) |
| BBC News | `sport` | Sports (1) |
| BBC News | `business` | Business (2) |
| BBC News | `tech` | Sci/Tech (3) |
| HuffPost | `WORLD NEWS` | World (0) |

Các nhãn BBC `politics`, `entertainment` và các nhãn HuffPost khác không được
đưa vào. Pipeline không tự đổi toàn bộ tin chính trị thành World.

## Mục đích sử dụng

Bộ dữ liệu phù hợp để:

- huấn luyện và so sánh mô hình phân loại bốn chủ đề AG News;
- nghiên cứu ảnh hưởng của việc bổ sung dữ liệu khác nguồn;
- kiểm tra pipeline làm sạch, chống trùng và bảo vệ tập đánh giá.

Bộ dữ liệu không được thiết kế cho phân loại tin tiếng Việt, phát hiện tin giả,
phân tích cảm xúc hoặc dự đoán các chủ đề ngoài bốn nhãn trên.

## Giới hạn

- BBC, HuffPost và AG News đến từ các giai đoạn và phong cách biên tập khác
  nhau. Mô hình có thể học dấu hiệu của nguồn thay vì chỉ học chủ đề.
- World chỉ được bổ sung từ HuffPost, còn ba lớp kia chỉ được bổ sung từ BBC.
  Số mẫu bổ sung nhỏ giúp giảm rủi ro này nhưng không loại bỏ hoàn toàn.
- File BBC không tách riêng tiêu đề. Pipeline dùng 16 từ đầu làm
  `title_clean`, phần còn lại làm `description_clean`, còn cột `text` vẫn giữ
  toàn bộ bài đã làm sạch.
- Một số bài BBC dài hơn giới hạn của BERT. Với `max_length=128`, tỷ lệ toàn bộ
  dữ liệu vượt giới hạn là 1%.
- Pipeline mới loại dữ liệu bổ sung trùng với tập đánh giá. Nó giữ nguyên tập
  AG News nền, vì vậy tình trạng gần trùng vốn có giữa AG News train và test
  không được thay đổi trong phiên bản này.

## Trạng thái chất lượng

Báo cáo tự động hiện có trạng thái `PASS` và sẵn sàng cho huấn luyện kỹ thuật.
Trước khi công bố hoặc chia sẻ phiên bản dữ liệu, cần đọc mẫu tại
`reports/multisource/manual_review_samples.csv` để kiểm tra nhãn bằng mắt.

## Lưu ý về giấy phép

Các bài báo vẫn có thể chịu bản quyền của đơn vị xuất bản ban đầu. Cần kiểm tra
điều khoản trên trang nguồn Kaggle trước khi phân phối lại dữ liệu hoặc sử dụng
ngoài phạm vi học tập, nghiên cứu. Repo này chỉ ghi lại nguồn và quy trình xử
lý; không cấp lại quyền đối với nội dung bài báo.
