#!/usr/bin/env python3

import os
import cv2
import numpy as np
import onnxruntime as ort
from gi.repository import Gst, GLib

# Khởi tạo GStreamer
Gst.init(None)

# Đường dẫn thư mục ảnh
IMAGE_FOLDER = "/home/aa/Code_Space/New/ai_gsstream/demo_images"

# Đường dẫn mô hình ONNX
MODEL_PATH = "/home/aa/Code_Space/New/ai_gsstream/bulldog_sumtn_v22_256_e59_a100.onnx"

# Tải mô hình ONNX
ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Tên input của mô hình
input_name = ort_session.get_inputs()[0].name
route_name = ort_session.get_inputs()[1].name

# Hàm preprocess ảnh
def preprocess_image(frame, shape):
    resized = cv2.resize(frame, (shape[3], shape[2]))  # Resize to model input size
    normalized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0  # Normalize and transpose to CHW
    return np.expand_dims(normalized, axis=0)  # Add batch dimension

# GStreamer callback để xử lý frame mới
def on_new_sample(sink, user_data):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)

    if success:
        # Đọc dữ liệu ảnh từ buffer
        data = np.ndarray(
            (user_data['height'], user_data['width'], 3),
            buffer=map_info.data,
            dtype=np.uint8
        )

        # Preprocess và chạy infer
        input_shape = ort_session.get_inputs()[0].shape
        processed_frame = preprocess_image(data, input_shape)

        # Tạo input giả lập cho route map
        routed_map = np.random.rand(1, 3, 128, 128).astype(np.float32)
        ort_inputs = {input_name: processed_frame, route_name: routed_map}

        # Chạy infer
        output = ort_session.run(None, ort_inputs)
        print(f"Output Shape: {output[0].shape}")

        buf.unmap(map_info)
    return Gst.FlowReturn.OK

# Khởi tạo pipeline từ ảnh
def start_image_stream(folder_path):
    # Tạo danh sách ảnh
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise RuntimeError("Không tìm thấy ảnh trong thư mục.")

    # Tạo video từ ảnh
    pipeline_description = f"filesrc location={images[0]} ! decodebin ! videoconvert ! appsink emit-signals=True name=appsink"
    pipeline = Gst.parse_launch(pipeline_description)

    appsink = pipeline.get_by_name("appsink")
    if not appsink:
        raise RuntimeError("Không tìm thấy appsink trong pipeline.")

    # Kết nối callback
    appsink.connect("new-sample", on_new_sample, {"width": 640, "height": 360})

    # Khởi động pipeline
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("Dừng pipeline.")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    start_image_stream(IMAGE_FOLDER)
