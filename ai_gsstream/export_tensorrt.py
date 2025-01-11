#!/usr/bin/env python3

# onnxruntime == 1.16.0

import onnxruntime as ort
import numpy as np
import time

MODEL_PATH = '/home/agx/ai_ws/src/behaviour_cloning/scripts/e2e_nav_3cams_n18_e119_rtx4090.onnx'

ort_option = {'device_id': 0,
              'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 3 GB
              'trt_engine_cache_enable': True,
              'trt_fp16_enable': True
              }
ort_session = ort.InferenceSession(MODEL_PATH, providers=[
        ('TensorrtExecutionProvider', ort_option),
        ])

input_name = ort_session.get_inputs()[0].name
route_name = ort_session.get_inputs()[1].name

image = np.random.rand(1, 3, 180, 820).astype(np.float32)
routed_map = np.random.rand(1, 3, 128, 128).astype(np.float32)

for i in range(100):
    start_time = time.time()
    ort_inputs = {input_name: image, route_name: routed_map}
    output = ort_session.run(None, ort_inputs)
    fps = 1 / (time.time() - start_time)
    print(f'Shape of output: {output[0].shape}, FPS: {fps}')
