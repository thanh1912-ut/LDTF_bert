# Từ điển dữ liệu

Ba file Parquet cuối dùng cùng một schema. Mỗi hàng đại diện cho một bài báo.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `row_id` | chuỗi | Mã duy nhất trong pipeline, có tiền tố nguồn |
| `source_dataset` | chuỗi | Nguồn: `ag_news`, `bbc_news` hoặc `huffpost` |
| `source_id` | chuỗi | Mã hàng hoặc liên kết ở nguồn ban đầu |
| `source_split` | chuỗi | Vai trò ban đầu: train, validation, test hoặc external |
| `original_label` | chuỗi | Nhãn trước khi ánh xạ về bốn lớp chung |
| `source_category` | chuỗi | Tên chuyên mục tại nguồn |
| `title_raw` | chuỗi | Tiêu đề trước khi làm sạch |
| `description_raw` | chuỗi | Mô tả hoặc nội dung trước khi làm sạch |
| `text_raw` | chuỗi | Văn bản đầu vào đầy đủ trước khi làm sạch |
| `label` | số nguyên 64 bit | Nhãn dùng để huấn luyện, thuộc khoảng 0 đến 3 |
| `label_name` | chuỗi | Tên nhãn: World, Sports, Business hoặc Sci/Tech |
| `title_clean` | chuỗi | Tiêu đề sau khi làm sạch |
| `description_clean` | chuỗi | Mô tả hoặc nội dung sau khi làm sạch |
| `text` | chuỗi | Văn bản model đọc khi `TEXT_ENCODING="single"` |
| `text_hash` | chuỗi | SHA-256 của văn bản chuẩn hóa, dùng để phát hiện trùng |

## Quan hệ giữa các cột văn bản

`raw` nghĩa là dữ liệu được giữ gần với nguồn để truy vết. Các cột `clean` đã
được chuẩn hóa Unicode, giải mã HTML, bỏ thẻ HTML và thu gọn khoảng trắng.

Model hiện dùng chế độ `single`, vì vậy đầu vào chính là cột `text`. Hai cột
`title_clean` và `description_clean` vẫn được lưu để có thể chạy thí nghiệm
sentence-pair mà không phải xử lý lại dữ liệu.

Với AG News và HuffPost, tiêu đề và mô tả có sẵn. Với BBC, file nguồn chỉ có
`category` và `text`; pipeline tạo tiêu đề kỹ thuật từ 16 từ đầu, rồi đặt phần
còn lại vào mô tả. Cột `text` của BBC vẫn chứa toàn bộ văn bản, không nối lặp
tiêu đề vừa tạo.

## Quy ước ID

Ví dụ:

```text
ag_news:train_000123
bbc_news:000456
huffpost:012345
```

`row_id` dùng cho manifest và kiểm tra trùng giữa các split. `source_id` dùng để
truy ngược về dữ liệu đầu vào. Không nên thay đổi hai cột này sau khi đã tạo một
phiên bản dataset.

## Điều kiện hợp lệ

- `row_id` không trùng trong cùng một split.
- `text` không rỗng.
- `label` chỉ nhận 0, 1, 2 hoặc 3.
- `label_name` phải khớp với `label`.
- `text_hash` không trùng trong train cuối.
- Validation và test không chứa mẫu BBC hoặc HuffPost.
