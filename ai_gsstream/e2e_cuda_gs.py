#!/usr/bin/env python3

# import gstreamer
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import time
import traceback
import cv2
import numpy as np
import onnxruntime as ort
import rospy
import numpy as np
import math
import threading, queue
from std_msgs.msg import String
from cargobot_msgs.msg import DriveState, Safety
from cargobot_msgs.msg import LaneFollowing
from cargobot_msgs.msg import GlobalPath
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import Imu
from pathlib import Path
from nav_msgs.msg import Odometry,Path
from collections import deque
from cargobot_msgs.msg import State,StateArray
from geometry_msgs.msg import PoseStamped

##### import pycuda and tensorrt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Initialize GStreamer
Gst.init(None)

##### Import Global Variables
AI_RATE = 60
INPUT_IMAGE_WIDTH = 820
INPUT_IMAGE_HEIGHT = 180
INPUT_ROUTE_WIDTH = 128
INPUT_ROUTE_HEIGHT = 128
TENSORRT_PATH = "/home/agx/ai_ws/src/behaviour_cloning/scripts/models/TensorrtExecutionProvider_TRTKernel_graph_main_graph_11891776995037849027_0_0_fp16_sm87.engine"
CUDA_KERNEL_PATH = "/home/agx/ai_ws/src/behaviour_cloning/scripts/alphaasimov.cu"
MTX_PATH = "/home/agx/ai_ws/src/behaviour_cloning/scripts/calib_matrix/intrinsic_matrix.npy"
DIST_PATH = "/home/agx/ai_ws/src/behaviour_cloning/scripts/calib_matrix/distortion_coefficients.npy"
CAMERA_FRONT_STREAM_URL = "rtsp://127.0.0.1:8554/camera_front"
CAMERA_LEFT_STREAM_URL = "rtsp://127.0.0.1:8554/camera_left"
CAMERA_RIGHT_STREAM_URL = "rtsp://127.0.0.1:8554/camera_right"

##### Cuda Wrapper
with open(CUDA_KERNEL_PATH, "r") as f:
    kernel_code = f.read()

cuda_mod = SourceModule(kernel_code)

class SteeringPredictor:
    def __init__(self, total_camera=4):
        self.lane_follow_ready = 1
        self.drive_mode = 2  # Drive mode:   0~RC  1~Tele  2~AI
        self.image_bridge = CvBridge()
        self.front_image = None
        self.back_image = None
        self.left_image = None
        self.right_image = None
        self.map_bridge = CvBridge()
        self.image = None  # (np.random.rand(IMG_HEIGHT, IMG_WIDTH, 3) * 255).astype(np.uint8)
        self.imu = None  # np.random.randn(1, 4).astype(np.float32)
        self.global_path = None  # np.random.randn(1, 1, 128, 128).astype(np.float32)
        self.yaw = None  # np.random.randn(1, 1).astype(np.float32)
        self.gps_error = None  # np.random.randn(1, 1).astype(np.float32)
        self.routed_map = None
        self.time = 0
        self.linear_x = 0
        self.angular_z = 0
        self.old_time = 0
        self.old_seq = 0
        self.seq = 0
        self.image_flag = False
        self.map_flag = False
        self.map_time = 0
        self.road_type = 0
        self.road_width = 0
        self.high_cmd = 0
        self.front_limit_speed = 1
        self.rear_limit_speed = -1
        self.contermet = 0

        self.image_queue = queue.Queue()
        self.received_images = {}

        self.topics = ['/camera_left/camera/image_raw/compressed',
                '/camera_front/camera/image_raw/compressed',
                '/camera_right/camera/image_raw/compressed',
                '/camera_back/camera/image_raw/compressed']

        self.total_camera = total_camera
        self.last_lane_following = LaneFollowing()

    def drive_state_callback(self, data):
        # Get Drive mode state from Drive Mode node
        drive_state = data
        try:
            self.drive_mode = drive_state.drive_mode_state
        except Exception as e:
            print(e)

    def odom_callback(self,data):
        self.linear_x = data.twist.twist.linear.x
        self.angular_z = data.twist.twist.linear.z

    def routed_map_callback(self, msg):
        try:
            if msg.encoding == '8UC3':
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == '8UC1':
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            elif msg.encoding == '8UC4':
                self.routed_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                self.routed_map = cv2.cvtColor(self.route_name, cv2.COLOR_RGBA2BGR)
            else:
                rospy.logerr(f"Unsupported encoding: {msg.encoding}")
                return

            self.map_flag = True
            self.map_time = time.time()

        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to OpenCV Image: {e}")

########### TensorRT functions
def cal_grid(width, height, block):
    dx, mx = divmod(width, block[0])
    dy, my = divmod(height, block[1])
    return ((dx + (mx>0)),(dy + (my>0)), 1)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers for inputs and outputs
def allocate_buffers(engine):
    h_inputs = []
    h_outputs = []
    d_inputs = []
    d_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            h_inputs.append(host_mem)
            d_inputs.append(device_mem)
        else:
            h_outputs.append(host_mem)
            h_outputs.append(device_mem)
    return h_inputs, h_outputs, d_inputs, d_outputs, bindings, stream
###########

########### GSTREAMER FUNCTIONS
def on_new_sample(sink, user_data):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()

    # Extract video frame data
    width = caps.get_structure(0).get_value("width")
    height = caps.get_structure(0).get_value("height")
    
    success, map_info = buf.map(Gst.MapFlags.READ)

    if success:
        data = np.ndarray((height, width, 3), buffer=map_info.data, dtype=np.uint8)
        user_data['frame'] = data
        buf.unmap(map_info)

    return Gst.FlowReturn.OK

def start_rtsp_stream(rtsp_url, app_name, user_data):
    pipeline = Gst.parse_launch(f"nvurisrcbin uri={rtsp_url} ! nvvidconv ! nvvideoconvert ! videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=True sync=False name={app_name}")

    appsink = pipeline.get_by_name(app_name)
    if not appsink:
        print(f"Failed to get appsink from pipeline for {rtsp_url}")
        return
    appsink.connect("new-sample", on_new_sample, user_data)

    # Start playing
    pipeline.set_state(Gst.State.PLAYING)

    try:
        # Run the loop
        user_data['loop'].run()
    except:
        pass

    # Clean up
    pipeline.set_state(Gst.State.NULL)

########## Main function ##########
def main():

    # Init GStream 
    rtsp_urls = [
        CAMERA_LEFT_STREAM_URL,
        CAMERA_FRONT_STREAM_URL,
        CAMERA_RIGHT_STREAM_URL
    ]

    app_names = ["appsink_left", "appsink_front", "appsink_right"]

    loops = [GLib.MainLoop() for _ in rtsp_urls]
    user_data_list = [{'loop': loop, 'frame': None} for loop in loops]

    threads = []
    for index, (rtsp_url, user_data) in enumerate(zip(rtsp_urls, user_data_list)):
        thread = threading.Thread(target=start_rtsp_stream, args=(rtsp_url, app_names[index], user_data))
        threads.append(thread)
        thread.start()

    # Alpha Asimov AD

    steering_predictor = SteeringPredictor(total_camera=3)
    rospy.init_node('E2E_Undistorted_Version')
    rospy.Subscriber("/drive_state", DriveState, steering_predictor.drive_state_callback)
    rospy.Subscriber("/routed_map", Image, steering_predictor.routed_map_callback)
    rospy.Subscriber("/odom", Odometry, steering_predictor.odom_callback)
    path_pub = rospy.Publisher("path_pred",Path, queue_size=1, tcp_nodelay=True)

    lane_follow_msg = LaneFollowing()
    pub_lane_follow_cmd = rospy.Publisher('/lane_follow_cmd', LaneFollowing, queue_size=1)

    trajectory = Path()
    trajectory.header.frame_id = "map"
    trajectory_pub = rospy.Publisher('/trajectory', Path, queue_size=1)    
    
    rate = rospy.Rate(AI_RATE) 

    rospy.loginfo("Subscribed to all topics")
    rospy.loginfo(f"Autopilot Sytem Rate: {AI_RATE} HZ")
    rospy.loginfo(f"Image Input Size: {INPUT_IMAGE_WIDTH} x {INPUT_IMAGE_HEIGHT}")
    rospy.loginfo(f"Route Input Size: {INPUT_ROUTE_WIDTH} x {INPUT_ROUTE_HEIGHT}")

    # Load Camera Matrix and Distortion Coefficients
    try:
        mtx = np.load(MTX_PATH).astype(np.float32)
        dist = np.load(DIST_PATH).astype(np.float32)
        rospy.loginfo(f"Loading Camera Matrix from: {MTX_PATH}")
        rospy.loginfo(f"Loading Distortion Coefficients from: {DIST_PATH}")
    except Exception as e:
        rospy.logerr(f"Error loading camera matrix: {e}")
        return

    # Load TensorRT model
    try:
        engine = load_engine(TENSORRT_PATH)
        context = engine.create_execution_context()
        rospy.loginfo(f"Loading TensorRT model from: {TENSORRT_PATH}")
    except Exception as e:
        rospy.logerr(f"Error loading TensorRT model: {e}")
        return

    # Allocate buffers
    h_inputs, h_outputs, d_inputs, d_outputs, bindings, stream = allocate_buffers(engine)

    # Prepare input data
    ratio_crop = int((420 / 1920) * 640)  # 140
    width_left_img = 640 - ratio_crop
    blur_pixel = int(width_left_img / 6)
    blur_intensity = 0.3
    block = (32, 32, 1)

    rospy.loginfo("Allocate buffers and prepare input data")
    rospy.loginfo(f"Ratio Crop: {ratio_crop}")
    rospy.loginfo(f"Width Left Image: {width_left_img}")
    rospy.loginfo(f"Blur Pixel: {blur_pixel}")
    rospy.loginfo(f"Blur Intensity: {blur_intensity}")
    rospy.loginfo(f"CUDA Block: {block}")

    # Get cuda kernels
    undistort_kernel = cuda_mod.get_function("undistortImageKernel")
    resize_kernel = cuda_mod.get_function("resizeImageKernel")
    blur_partial_kernel = cuda_mod.get_function("blurPartialImageKernel")
    crop_kernel = cuda_mod.get_function("cropImageKernel")
    concat_kernel = cuda_mod.get_function("concatImagesKernel")

    left_image_cropped = np.random.rand(1, 3, 360, 640-ratio_crop).astype(np.float32).ravel()
    right_image_cropped = np.random.rand(1, 3, 360, 640-ratio_crop).astype(np.float32).ravel()
    concat_image = np.random.rand(1, 3, 360, 1640).astype(np.float32).ravel()
    routed_map_reszied = np.random.rand(1, 3, 128, 128).astype(np.float32).ravel()

    # Allocate GPU memory
    input_nbytes = 3 * 640 * 360 * np.dtype(np.float32).itemsize
    left_image_gpu = cuda.mem_alloc(input_nbytes)
    left_image_gpu_undistorted = cuda.mem_alloc(input_nbytes)
    left_image_gpu_cropped = cuda.mem_alloc(left_image_cropped.nbytes)
    center_image_gpu = cuda.mem_alloc(input_nbytes)
    center_image_gpu_undistorted = cuda.mem_alloc(input_nbytes)
    right_image_gpu = cuda.mem_alloc(input_nbytes)
    right_image_gpu_undistorted = cuda.mem_alloc(input_nbytes)
    right_image_gpu_cropped = cuda.mem_alloc(right_image_cropped.nbytes)
    concat_image_gpu = cuda.mem_alloc(concat_image.nbytes)

    try:
        while not rospy.is_shutdown():
            start_time = time.time()
            # stop AI if in RC mode (0) or joystick mode (3)
            if steering_predictor.drive_mode == 0 or steering_predictor.drive_mode == 3: 
                rospy.logerr("-------------- Not in AI mode -------------")
                time.sleep(1)
                continue        

            routed_map = steering_predictor.routed_map

            if not steering_predictor.map_flag:
                if (time.time() - steering_predictor.map_time) > 0.5:
                    rospy.logerr(f"------------- Waiting for routed map here!")
                    time.sleep(1)    
                    continue
            
            frames = [user_data['frame'] for user_data in user_data_list]
            if all(frame is not None for frame in frames): # all frames are ready
                left_image, center_image, right_image = frames

                # Prepare gpu buffers
                left_image = left_image.astype(np.float32).ravel()
                center_image = center_image.astype(np.float32).ravel()
                right_image = right_image.astype(np.float32).ravel()

                cuda.memcpy_htod_async(left_image_gpu, left_image, stream)
                cuda.memcpy_htod_async(center_image_gpu, center_image, stream)
                cuda.memcpy_htod_async(right_image_gpu, right_image, stream)

                # Run cuda kernels
                undistort_kernel(center_image_gpu, center_image_gpu_undistorted, np.int32(640), np.int32(360), mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2], dist[0], dist[1], dist[2], dist[3], dist[4], block=block, grid=cal_grid(640, 360, block))
                undistort_kernel(left_image_gpu, left_image_gpu_undistorted, np.int32(640), np.int32(360), mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2], dist[0], dist[1], dist[2], dist[3], dist[4], block=block, grid=cal_grid(640, 360, block))
                undistort_kernel(right_image_gpu, right_image_gpu_undistorted, np.int32(640), np.int32(360), mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2], dist[0], dist[1], dist[2], dist[3], dist[4], block=block, grid=cal_grid(640, 360, block))

                crop_kernel(left_image_gpu_undistorted, left_image_gpu_cropped, np.int32(0), np.int32(0), np.int32(640), np.int32(640-ratio_crop), np.int32(360), block=block, grid=cal_grid(640, 360, block))
                crop_kernel(right_image_gpu_undistorted, right_image_gpu_cropped, np.int32(ratio_crop), np.int32(0), np.int32(640), np.int32(640-ratio_crop), np.int32(360), block=block, grid=cal_grid(640, 360, block))

                blur_partial_kernel(left_image_gpu_cropped, np.int32(500), np.int32(360), np.int32(500 - blur_pixel),  np.int32(500), np.int32(0), np.int32(360), np.float32(blur_intensity), block=block, grid=cal_grid(500, 360, block))
                blur_partial_kernel(right_image_gpu_cropped, np.int32(500), np.int32(360), np.int32(0), np.int32(blur_pixel), np.int32(0), np.int32(360), np.float32(blur_intensity), block=block, grid=cal_grid(500, 360, block))

                concat_kernel(left_image_gpu_cropped, center_image_gpu_undistorted, right_image_gpu_cropped, concat_image_gpu, np.int32(500), np.int32(640), np.int32(500), np.int32(360), np.int32(1640), np.int32(360), block=block, grid=cal_grid(1640, 360, block))
                resize_kernel(concat_image_gpu, np.int32(1640), np.int32(360), d_inputs[0], np.int32(820), np.int32(180), np.float32(360 / 180), np.float32(1640 / 820), block=block, grid=cal_grid(1640, 360, block))

                # for debugging
                # resize_new = np.zeros((3, 180, 820)).astype(np.float32).ravel()
                # cuda.memcpy_dtoh_async(resize_new, d_inputs[0], stream)
                # resize_new = np.reshape(resize_new, (3, 180, 820)).transpose(1, 2, 0).astype(np.uint8) # CHW -> HWC 
                # resize_new = cv2.cvtColor(resize_new, cv2.COLOR_RGB2BGR) # RGB -> BGR 
                # cv2.imwrite("/home/agx/ai_ws/src/behaviour_cloning/scripts/resize.jpg", resize_new)

                # for debugging
                # routed_map = cv2.imread('/home/agx/ai_ws/src/behaviour_cloning/scripts/demo_images/routed_map_2024_06_30_11_17_03_524.jpg')
            
                routed_map = routed_map[80-64:80+64, 80-64:80+64] # crop 160x160 -> 128x128 
                routed_map = cv2.cvtColor(routed_map, cv2.COLOR_BGR2RGB)
                routed_map = routed_map.transpose(2, 0, 1).astype(np.float32)

                # copy routed map to GPU
                np.copyto(h_inputs[1], routed_map.flatten())
                cuda.memcpy_htod_async(d_inputs[1], h_inputs[1], stream)

                # Run inference
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

                # Transfer predictions back from the GPU
                cuda.memcpy_dtoh_async(h_outputs[0], bindings[2], stream)

                # Synchronize the stream
                stream.synchronize()

                # Reshape output to expected shape
                output = h_outputs[0].reshape(1, 50, 2)

                FPS = 1 / (time.time() - start_time)

                linear_vel = output[0][0][0]
                angular_vel = output[0][0][1] #+ ANGULAR_OFFSET

                lane_follow_msg.lane_follow_ready = 1
                lane_follow_msg.lane_follow_vel.linear.x = linear_vel
                lane_follow_msg.lane_follow_vel.angular.z = angular_vel
                pub_lane_follow_cmd.publish(lane_follow_msg)

                # steering_predictor.last_lane_following = lane_follow_msg

                rospy.loginfo(f"LV and AV: {output[0][0][0]} | {output[0][0][1]} | FPS: {FPS:.2f}")

            else:
                rospy.logerr("-------------- Waiting for all frames -------------")
                time.sleep(1)

        rate.sleep()

    finally:
        for loop in loops:
            loop.quit()
        for thread in threads:
            thread.join()
        
        # release memory
        del context
        del engine

        # release gpu memory
        left_image_gpu.free()
        left_image_gpu_undistorted.free()
        left_image_gpu_cropped.free()
        center_image_gpu.free()
        center_image_gpu_undistorted.free()
        right_image_gpu.free()
        right_image_gpu_undistorted.free()
        right_image_gpu_cropped.free()
        concat_image_gpu.free()

        rospy.logwarn("----------------- End of AI mode -----------------")
        rospy.logwarn("TensorRT engine and context destroyed")
        rospy.logwarn("GStreamer pipelines stopped")
        rospy.logwarn("CUDA memory released")
        rospy.logwarn("--------------------------------------------------")

if __name__ == "__main__":
    main()