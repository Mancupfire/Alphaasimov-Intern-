import sys
import pyds
import gi
import numpy as np
from gi.repository import Gst, GObject
from pathlib import Path
import time

# Khởi tạo GStreamer
Gst.init(None)

# Path dẫn đến model .ONNX
onnx_model_path = "/model.onnx"

# Khởi tạo DeepStream pipeline
def create_pipeline(camera_source, onnx_model_path):
    pipeline = Gst.Pipeline.new("pipeline")
    
    source = Gst.ElementFactory.make("nvarguscamerasrc", "camera-source")
    if not source:
        print("Không thể tạo nguồn video từ camera.")
        return None

    convert = Gst.ElementFactory.make("nvvideoconvert", "converter")
    if not convert:
        print("Không thể tạo bộ chuyển đổi video.")
        return None
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinfer")
    if not pgie:
        print("Không thể tạo nvinfer element.")
        return None
    
    pgie.set_property('config-file-path', onnx_model_path)  # Đảm bảo cấu hình mô hình đúng

    sink = Gst.ElementFactory.make("nveglglessink", "sink")
    if not sink:
        print("Không thể tạo bộ hiển thị video.")
        return None

    pipeline.add(source)
    pipeline.add(convert)
    pipeline.add(pgie)
    pipeline.add(sink)
    
    source.link(convert)
    convert.link(pgie)
    pgie.link(sink)

    return pipeline

def run_pipeline(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline đang chạy...")

    loop = GObject.MainLoop()
    loop.run()

if __name__ == "__main__":
    camera_source = "nvarguscamerasrc" 
    
    pipeline = create_pipeline(camera_source, onnx_model_path)
    if pipeline:
        run_pipeline(pipeline)
