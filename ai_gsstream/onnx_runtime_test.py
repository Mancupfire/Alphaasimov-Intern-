#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import time
import cv2

# Model path
MODEL_PATH = '/home/aa/Code_Space/New/ai_gsstream/bulldog_sumtn_v22_256_e59_a100.onnx'

# Use CUDAExecutionProvider if GPU is available, otherwise fallback to CPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"ONNX model loaded successfully with providers: {ort_session.get_providers()}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    exit(1)

# Model input names
input_name = ort_session.get_inputs()[0].name  # Image input
route_name = ort_session.get_inputs()[1].name  # Route map input

# Function to preprocess images
def preprocess_image(frame, shape):
    resized = cv2.resize(frame, (shape[3], shape[2]))  # Resize to model input size
    normalized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0  # Normalize and transpose to CHW
    return np.expand_dims(normalized, axis=0)  # Add batch dimension

# Load or simulate input data
image_shape = ort_session.get_inputs()[0].shape
route_shape = ort_session.get_inputs()[1].shape

# Simulated data or camera input
def get_camera_input(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera: {camera_index}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame from camera: {camera_index}")
    return frame

def generate_route_map():
    return np.random.rand(1, 3, route_shape[2], route_shape[3]).astype(np.float32)

# Run inference
try:
    for i in range(10):  # Run inference multiple times for testing
        start_time = time.time()

        # Get input data (replace with real camera frames in production)
        image_frame = get_camera_input(0)  # Use camera 0 for testing
        processed_image = preprocess_image(image_frame, image_shape)
        routed_map = generate_route_map()  # Simulate route map input

        # Prepare ONNX inputs
        ort_inputs = {input_name: processed_image, route_name: routed_map}

        # Perform inference
        output = ort_session.run(None, ort_inputs)

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        print(f"Output Shape: {output[0].shape}, FPS: {fps:.2f}")
except Exception as e:
    print(f"Error during inference: {e}")
