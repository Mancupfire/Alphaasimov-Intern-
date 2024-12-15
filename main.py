import sys
import pyds
import gi
import numpy as np
from gi.repository import Gst, GObject
from pathlib import Path
import time
from huggingface_hub import login
from transformers import AutoTokenizer
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# Khởi tạo GStreamer
Gst.init(None)

local_onnx_path = hf_hub_download(repo_id="microsoft/Phi-3-vision-128k-instruct-onnx", filename="model.onnx")
print("ONNX model downloaded to:", local_onnx_path)

session = ort.InferenceSession(local_onnx_path)

def create_pipeline(camera_source, config_file_path):
    pipeline = Gst.Pipeline.new("pipeline")

    source = Gst.ElementFactory.make(camera_source, "video-source")
    if not source:
        print(f"Không thể tạo nguồn video từ {camera_source}.")
        return None
    
    if camera_source == "v4l2src":
        source.set_property("device", "/dev/video0")  # Webcam
    elif camera_source == "filesrc":
        source.set_property("location", "/path/to/video.mp4")  # Video file

    convert = Gst.ElementFactory.make("nvvideoconvert", "converter")
    if not convert:
        print("Không thể tạo bộ chuyển đổi video.")
        return None

    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinfer")
    if not pgie:
        print("Không thể tạo nvinfer element.")
        return None
    pgie.set_property('config-file-path', config_file_path)

    sink = Gst.ElementFactory.make("nveglglessink", "sink")
    if not sink:
        print("Không thể tạo bộ hiển thị video.")
        return None

    pipeline.add(source)
    pipeline.add(convert)
    pipeline.add(pgie)
    pipeline.add(sink)

    try:
        source.link(convert)
        convert.link(pgie)
        pgie.link(sink)
    except Exception as e:
        print("Lỗi khi liên kết pipeline:", e)
        return None

    return pipeline

def run_pipeline(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline đang chạy...")

    loop = GObject.MainLoop()
    loop.run()

if __name__ == "__main__":
    camera_source = "/dev/video0"
    config_file_path = "/path/to/Alphaasimov/config_file.txt"
    
    pipeline = create_pipeline(camera_source, config_file_path)
    if pipeline:
        run_pipeline(pipeline)
    else:
        print("Pipeline không thể khởi tạo.")
