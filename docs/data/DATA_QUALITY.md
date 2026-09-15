# Chất lượng dữ liệu

## Trạng thái hiện tại

Phiên bản `multisource-v1` có trạng thái `PASS` cho toàn bộ kiểm tra kỹ thuật tự
động và `ready_for_training=true`.

`PASS` xác nhận cấu trúc, dữ liệu rỗng, nhãn, trùng lặp, split, tokenizer và
manifest đạt điều kiện đã cấu hình. Nó không khẳng định mọi nhãn nguồn đều đúng
về mặt biên tập.

## Kết quả làm sạch

Pipeline kiểm tra 132.027 hàng trước khi lấy mẫu cuối:

| Kiểm tra | Số lỗi còn lại |
| --- | ---: |
| Văn bản rỗng | 0 |
| Thẻ HTML | 0 |
| Numeric HTML entity | 0 |
| Ký tự backslash lỗi | 0 |

Không có hàng nào bị loại do rỗng sau khi làm sạch.

## Kết quả loại trùng

| Loại | Số hàng bị loại |
| --- | ---: |
| Trùng hoàn toàn, cùng nhãn | 72 |
| Gần trùng, cùng nhãn | 40 |
| Trùng với validation/test do dữ liệu bổ sung | 0 |

Train cuối không còn `text_hash` trùng. Dữ liệu bổ sung không tạo overlap mới
với validation hoặc test ở ngưỡng gần trùng 0,90.

## Phân bố dữ liệu

| Nhãn | Train cuối |
| --- | ---: |
| World | 27.298 |
| Sports | 27.289 |
| Business | 27.238 |
| Sci/Tech | 27.206 |

Phần bổ sung cân bằng tuyệt đối ở mức 324 mẫu cho mỗi nhãn. Chênh lệch nhỏ của
train cuối đến từ AG News nền.

## Độ dài token

Tokenizer: `bert-base-uncased`.

| Thống kê | Giá trị |
| --- | ---: |
| Số hàng kiểm tra | 128.602 |
| Trung bình | 53,502 token |
| Trung vị | 49 token |
| P95 | 74 token |
| P99 | 128 token |
| Lớn nhất | 3.424 token |
| Tỷ lệ vượt 128 token | 1% |

Ngưỡng chấp nhận trong cấu hình là 5%, nên kiểm tra truncation đạt PASS. Các
bài rất dài chủ yếu đến từ BBC; model chỉ đọc phần đầu sau khi cắt ở 128 token.

## Bảo vệ tập đánh giá

Fingerprint validation:

```text
8ac06001d45a75f978a6734906fb6b31b06c4d0772638ebabab9359e24edc6d8
```

Fingerprint test:

```text
3294bf262ac56d4a5ecfabe215a03875e47d662d0e42e36b62803ff6232f2d1e
```

Hai fingerprint được giữ nguyên sau pipeline. Test vẫn là tập đánh giá chính
thức và không được dùng để chọn mô hình, epoch hoặc siêu tham số.

## Kiểm tra nhãn bằng mắt

File `reports/multisource/manual_review_samples.csv` chứa 25 mẫu cho từng cặp
nguồn và nhãn, tổng cộng 100 hàng. Cần kiểm tra các điểm sau:

- nội dung có đúng chủ đề đã gắn không;
- bài Business có bị nhầm với Sci/Tech không;
- bài World có thực sự là tin quốc tế không;
- phần đầu bài BBC có đủ thông tin để model phân loại không.

Sau khi duyệt, nên ghi người duyệt, ngày duyệt, số lỗi và quyết định chấp nhận
vào một phiên bản báo cáo mới. Không chỉnh trực tiếp file mẫu do pipeline tạo.

## Rủi ro còn lại

- Source bias do World bổ sung từ HuffPost, ba nhãn còn lại từ BBC.
- Khác biệt thời gian và văn phong giữa ba nguồn.
- Nhãn do nguồn cung cấp có thể chứa trường hợp biên hoặc sai nhãn.
- AG News nền được giữ nguyên, bao gồm các hạn chế trùng lặp đã tồn tại từ
  trước khi bổ sung BBC/HuffPost.
