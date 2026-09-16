# BẮT ĐẦU Ở ĐÂY

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
