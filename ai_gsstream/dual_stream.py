#!/usr/bin/env python3

import cv2
import numpy as np
import onnxruntime as ort
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# ONNX Model path
MODEL_PATH = '/home/aa/Code_Space/New/ai_gsstream/bulldog_sumtn_v22_256_e59_a100.onnx'

ort_session = ort.InferenceSession(MODEL_PATH, providers=[
    'CUDAExecutionProvider',  
    'CPUExecutionProvider'    
])

input_name = ort_session.get_inputs()[0].name

def preprocess_frame(frame, input_shape):
    frame = cv2.resize(frame, (input_shape[3], input_shape[2]))  
    frame = frame.astype(np.float32).transpose(2, 0, 1)  
    frame = np.expand_dims(frame, axis=0) 
    return frame

def on_new_sample(sink, user_data):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)

    if success:
        frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        # Preprocess frame
        input_shape = ort_session.get_inputs()[0].shape
        preprocessed_frame = preprocess_frame(frame, input_shape)

        # Run inference
        start_time = time.time()
        output = ort_session.run(None, {input_name: preprocessed_frame})
        fps = 1 / (time.time() - start_time)

        print(f'Output: {output}, FPS: {fps:.2f}')
        buf.unmap(map_info)

    return Gst.FlowReturn.OK

def start_rtsp_inference(rtsp_url):
    pipeline = Gst.parse_launch(f"rtspsrc location={rtsp_url} ! decodebin ! videoconvert ! appsink emit-signals=True name=appsink")
    appsink = pipeline.get_by_name("appsink")
    appsink.connect("new-sample", on_new_sample, None)

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop = GLib.MainLoop()
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    RTSP_URL = "rtsp://127.0.0.1:8554"  
    start_rtsp_inference(RTSP_URL)
