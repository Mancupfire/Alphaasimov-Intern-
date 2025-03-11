# OD Groundtruth Generator
Dự án **OD Groundtruth Generator** là một công cụ Python dùng để tạo groundtruth cho bài toán Object Detection (OD) dựa trên mô hình YOLOv8 từ thư viện Ultralytics. Công cụ này thực hiện:

- Duyệt qua các hình ảnh trong một thư mục dữ liệu.
- Sử dụng mô hình YOLOv8 (yolov8s.pt) để phát hiện các đối tượng.
- Lưu lại hình ảnh đã được annotate vào thư mục đầu ra.
- Tạo file JSON chứa thông tin các đối tượng (bao gồm `class_id`, `class_name` và bounding box) chỉ với các đối tượng thuộc danh mục xe cộ.

---

## Yêu Cầu
- **Python 3.7 trở lên**
- **Thư viện**:
  - [Ultralytics](https://github.com/ultralytics/ultralytics) – để sử dụng mô hình YOLOv8
  - [PyTorch](https://pytorch.org/) – xử lý mô hình trên GPU/CPU
- Các thư viện tích hợp: `os`, `json`

---

## Cấu Trúc Thư Mục
- **data_folder**  
  Thư mục chứa các hình ảnh cần xử lý (hỗ trợ định dạng `.jpg` và `.png`).
- **output_folder**  
  Thư mục lưu kết quả, bao gồm:
  - Ảnh đã được annotate (được lưu trong thư mục con `annotated`)
  - File JSON chứa thông tin groundtruth (định dạng: `image_name_gt.json`)

---

## Cài Đặt
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install ultralytics torch
