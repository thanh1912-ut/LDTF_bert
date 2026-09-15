# Cấu trúc dữ liệu

Thư mục này tách dữ liệu nguồn bất biến, dữ liệu trung gian và đầu ra dùng cho
model.

## Các thư mục

- `raw/ag_news`: CSV train và test AG News gốc;
- `raw/bbc_news`: file nén Kaggle và dữ liệu BBC đã giải nén;
- `raw/huffpost`: file nén Kaggle và dữ liệu HuffPost đã giải nén;
- `cache/huggingface`: tokenizer được lưu cục bộ;
- `standardized`: từng nguồn sau khi chuyển về schema chung;
- `merged`: dữ liệu đã gộp trước và sau khi loại trùng;
- `processed`: train, validation và test Parquet cuối.

Không chỉnh sửa trực tiếp file trong `raw`. Mọi bước xử lý phải ghi đầu ra mới
vào `standardized`, `merged` hoặc `processed`.

## Nguồn tải

- BBC articles fulltext and category:
  https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category
- News Category Dataset của HuffPost:
  https://www.kaggle.com/datasets/rmisra/news-category-dataset

Cần đọc điều khoản tại trang nguồn trước khi phân phối lại dữ liệu. Nội dung bài
báo có thể vẫn thuộc bản quyền của nhà xuất bản ban đầu.

## Đầu ra

```text
processed/
├── research_train.parquet
├── research_validation.parquet
└── research_test.parquet
```

Pipeline ghi kiểm tra chất lượng tại `reports/multisource` và thông tin tái lập
tại `manifests/multisource`. Validation và test AG News được giữ nguyên để kết
quả thí nghiệm có thể so sánh trực tiếp.
