# Hướng dẫn sử dụng hệ thống Image and Summarize Document Retrieval Application for Animals (CS406)

## 1. Cài đặt môi trường

- Yêu cầu Python 3.10+ và pip.
- Cài đặt các thư viện cần thiết:

```bash
cd src
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

- Dữ liệu ảnh động vật nằm trong thư mục `data/animals/` (mỗi loài 1 thư mục con).
- Dữ liệu phân đoạn cảnh vật: `data/seg/` và `data/seg_test/`.

## 3. Tiền xử lý & Tạo chỉ mục dữ liệu

- Tạo chỉ mục dữ liệu cho hệ thống retrieval:

```bash
python index_data.py
```

- Tạo dữ liệu tổng hợp (nếu cần):

```bash
python synthetic_data.py
```

## 4. Huấn luyện mô hình

- Huấn luyện classifier động vật:

```bash
python train_classifier.py
```

- Huấn luyện mô hình phân đoạn, detection, hoặc các mô hình khác (nếu có):

```bash
# Ví dụ, nếu có script riêng cho object detection
python object_detection.py
```

## 5. Đánh giá mô hình

```bash
python evaluate.py
```

## 6. Chạy server backend

- Chạy API backend (Flask FastAPI, v.v. — tuỳ cấu hình trong app.py):

```bash
python app.py
```

## 7. Giao diện frontend

- Mở file `frontend/index.html` hoặc `frontend/demo.html` trên trình duyệt để sử dụng giao diện demo.
- Đảm bảo backend đã chạy trước khi sử dụng frontend.

## 8. Truy vấn & truy xuất ảnh

- Upload ảnh truy vấn qua giao diện web hoặc gửi request tới API backend.
- Ảnh kết quả sẽ được trả về dựa trên pipeline retrieval đã xây dựng.

## 9. Một số script hữu ích

- `split_dataset.py`: Chia tập train/test.
- `predict.py`: Dự đoán nhãn cho ảnh mới.
- `kf_extractor.py`: Trích xuất đặc trưng (key features).

## 10. Lưu ý

- Các model đã train lưu ở thư mục `models/` hoặc cùng cấp với script.
- Nếu gặp lỗi thiếu thư viện, kiểm tra lại `requirements.txt` hoặc cài đặt bổ sung.
- Đường dẫn dữ liệu có thể cần chỉnh lại cho phù hợp với máy của bạn.

---

**Mọi thắc mắc vui lòng liên hệ chủ dự án hoặc xem chi tiết trong từng file script.**
