Dự án OD Groundtruth Generator là một công cụ Python dùng để tạo groundtruth cho bài toán Object Detection (OD) dựa trên mô hình YOLOv8 từ thư viện Ultralytics. Công cụ này thực hiện:

Duyệt qua các hình ảnh trong một thư mục dữ liệu.
Sử dụng mô hình YOLOv8 (yolov8s.pt) để phát hiện các đối tượng.
Lưu lại hình ảnh đã được annotate vào thư mục đầu ra.
Tạo file JSON chứa thông tin các đối tượng (bao gồm class_id, class_name và bounding box) chỉ với các đối tượng thuộc danh mục xe cộ.
Yêu Cầu
Python 3.7 trở lên
Thư viện:
Ultralytics – để sử dụng mô hình YOLOv8.
PyTorch – xử lý mô hình trên GPU/CPU.
Các thư viện tích hợp: os, json.
Cài đặt các thư viện cần thiết:

bash
Copy
pip install ultralytics torch
Cấu Trúc Thư Mục
data_folder:
Thư mục chứa các hình ảnh cần xử lý (hỗ trợ định dạng .jpg và .png).

output_folder:
Thư mục lưu kết quả, gồm:

Ảnh đã được annotate (được lưu trong thư mục con annotated).
File JSON chứa thông tin groundtruth (định dạng: image_name_gt.json).
Hướng Dẫn Sử Dụng
Cập nhật đường dẫn thư mục:

Mở file source code và chỉnh sửa các biến data_folder và output_folder sao cho phù hợp với cấu trúc trên máy của bạn:

python
Copy
data_folder = r"D:\data_use\2024_11_22_08_52_03\camera_back"
output_folder = r"D:\data_use\2024_11_22_08_52_03\camerabackk_Obj_detection"
Chạy Code:

Mở terminal hoặc command prompt, chuyển đến thư mục chứa file code và chạy lệnh:

bash
Copy
python your_script_name.py
Kết quả:

Ảnh đã được annotate sẽ được lưu trong thư mục output_folder tại thư mục con annotated.
File JSON chứa thông tin các đối tượng sẽ được lưu trong output_folder với định dạng tên file: image_name_gt.json.
Giải Thích Code
Khởi tạo mô hình:

python
Copy
model = YOLO("yolov8s.pt")
Sử dụng mô hình YOLOv8 yolov8s.pt để thực hiện dự đoán trên các hình ảnh.

Lọc các đối tượng xe cộ:

Chỉ xử lý các đối tượng có class_id thuộc danh mục xe cộ được định nghĩa trong vehicle_class_ids:

python
Copy
vehicle_class_ids = [1, 2, 3, 5, 7]
Tên các lớp đối tượng được lưu trong dictionary class_names.

Xử lý từng ảnh:

Hàm process_image sẽ:

Dự đoán các đối tượng có trong ảnh.
Lưu ảnh đã annotate vào thư mục đầu ra.
Trích xuất thông tin bounding box và class ID của các đối tượng thuộc danh mục xe cộ.
Tạo file JSON chứa thông tin chi tiết của các đối tượng.
Vòng lặp xử lý:

Code sẽ duyệt qua tất cả các ảnh trong data_folder có đuôi .jpg hoặc .png, xử lý và lưu kết quả tương ứng.

Tùy Chỉnh
Danh mục đối tượng:

Để xử lý các đối tượng khác hoặc thay đổi danh mục, bạn có thể chỉnh sửa:

Danh sách vehicle_class_ids.
Dictionary class_names.
Tham số dự đoán của mô hình:

Bạn có thể tùy chỉnh các tham số của hàm model.predict (như verbose, device, save, project, name) theo nhu cầu.
