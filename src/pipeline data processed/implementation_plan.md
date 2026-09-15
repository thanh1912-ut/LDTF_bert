# Trạng thái triển khai pipeline dữ liệu

Tài liệu này từng là kế hoạch cho pipeline AG News ban đầu. Pipeline đa nguồn
hiện đã được triển khai và chạy thành công, vì vậy thông tin thiết kế chính thức
được chuyển sang các tài liệu trong `docs/data`.

## Hạng mục đã hoàn thành

- tổ chức riêng dữ liệu raw, standardized, merged và processed;
- adapter cho AG News, BBC News và HuffPost;
- ánh xạ về bốn nhãn World, Sports, Business, Sci/Tech;
- làm sạch HTML, Unicode, escape và khoảng trắng;
- loại trùng hoàn toàn và gần trùng bằng MinHash/LSH;
- bảo vệ validation và test khỏi dữ liệu bổ sung bị trùng;
- lấy mẫu bổ sung cân bằng theo nhãn với seed cố định;
- phân tích độ dài `bert-base-uncased` và kiểm tra batch;
- xuất Parquet, checksum, manifest và báo cáo PASS;
- kiểm thử adapter, làm sạch, lấy mẫu và chống overlap.

## Kết quả phiên bản multisource-v1

| Tập | Số mẫu |
| --- | ---: |
| Train | 109.031 |
| Validation | 11.971 |
| Test | 7.600 |

Train được bổ sung 324 mẫu cho mỗi nhãn, tổng cộng 1.296 mẫu. Báo cáo cuối có
trạng thái `PASS` cho kiểm tra kỹ thuật tự động.

## Tài liệu tiếp tục sử dụng

- `../../docs/data/DATASET_CARD.md`;
- `../../docs/data/DATA_DICTIONARY.md`;
- `../../docs/data/PROCESSING_PIPELINE.md`;
- `../../docs/data/DATA_QUALITY.md`;
- `../../docs/data/REPRODUCIBILITY.md`.

Các thay đổi pipeline trong tương lai phải cập nhật cấu hình, tài liệu tương
ứng và tạo phiên bản dataset mới thay vì sửa im lặng `multisource-v1`.
