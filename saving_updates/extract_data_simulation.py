#!/usr/bin/env python2

import time
import datetime
import numpy as np
import os
import cv2
import csv
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, Joy, CompressedImage
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError
from cargobot_msgs.msg import RemoteControl, JoystickControl
from cargobot_msgs.msg import DriveState, DeliveryState
from cargobot_msgs.msg import Safety, Route
from collect_data.msg import GlobalPath, HighCmd
import argparse

LINEAR_MAX = 3.0  # m/s
ANGULAR_MAX = 1.0  # rad/s
LIN_VEL_LIMIT_DEFAULT = [-10, 100]
STEP = 0.1
AP_MODE = 2

# Command-line arguments
parser = argparse.ArgumentParser(description='Extract data from rosbag')
parser.add_argument('--bag', dest='bag_file', default='/2020_01_01', type=str, help='Name of the rosbag file')
parser.add_argument('--rate', dest='rate', default=1, type=int, help='Multiply rosbag rate record')
parser.add_argument('--fps', dest='fps', default=20, type=int, help='Frame rate of rosbag record')
args = parser.parse_args()


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z  # in radians


class LogData:
    def __init__(self, camera_name, topic_name, path_dir):
        self.camera_name = camera_name
        self.folder_name = args.bag_file.split('/')[-1][:10]
        self.logDir = os.path.join(path_dir, self.folder_name)
        os.makedirs(self.logDir, exist_ok=True)

        self.folder_name = args.bag_file.split('/')[-1][:-4]
        self.CSV_filename = f"{self.folder_name}.csv"
        self.imgDir = os.path.join(self.logDir, 'IMG')
        os.makedirs(self.imgDir, exist_ok=True)

        self.initialize_fields()
        rospy.init_node('collect_drive_data_node', anonymous=True)

        # Subscribing to ROS topics
        self.subscribe_topics(topic_name)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(args.rate * args.fps)
        self.threash_rate = args.fps * args.rate
        self.lin_vel_limit = LIN_VEL_LIMIT_DEFAULT

    def initialize_fields(self):
        self.imu_msg = Imu()
        self.yaw = -1.0
        self.gps_msg = NavSatFix()
        self.gps_error = -1.0
        self.sonar_right = -1
        self.lidar_front_right = -1
        self.sonar_front_center = -1
        self.lidar_front_center = -1
        self.lidar_front_left = -1
        self.sonar_left = -1
        self.sonar_rear_center = -1
        self.cv_image = None
        self.img_callback_flag = False
        self.enable_collect_data_rc = False
        self.enable_collect_data_joy = False
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.linear_vel_feedback = 0.0
        self.angular_vel_feedback = 0.0
        self.dir_high_cmd = 0.0
        self.mode = 0
        self.secs = 0
        self.nsecs = 0
        self.crr_time = 0
        self.old_time = 0
        self.counting = 0
        self.global_path_nearest = GlobalPath()
        self.route = []

    def subscribe_topics(self, topic_name):
        rospy.Subscriber('/delivery_state', DeliveryState, self.deliveryStateCallback)
        rospy.Subscriber("/ublox/fix", NavSatFix, self.gnssCallback)
        rospy.Subscriber("/imu/data", Imu, self.imuCallback)
        rospy.Subscriber("/safety_limit_speed", Safety, self.safetySensorCallback)
        rospy.Subscriber("/global_path_nearest", GlobalPath, self.global_path_Callback)
        rospy.Subscriber("/dir_high_cmd", HighCmd, self.dirHighCmdCallback)
        rospy.Subscriber("/joystick_control", JoystickControl, self.joystickCmdCallback)
        rospy.Subscriber("/drive_state", DriveState, self.driveStateCallback)
        rospy.Subscriber("/route", Route, self.route_Callback)
        rospy.Subscriber("/odom", Odometry, self.odomCallback)
        rospy.Subscriber(topic_name, CompressedImage, self.imageCallback)

    def deliveryStateCallback(self, data):
        self.delivery_state = data.delivery_state_monitor
        self.enable_collect_data_gps = self.delivery_state in [4, 5, 7, 8]

    def global_path_Callback(self, data):
        self.global_path_nearest = data.global_path

    def route_Callback(self, data):
        self.route = data.list_point

    def driveStateCallback(self, msg):
        self.drive_state = msg.drive_mode_state
        self.linear_vel = msg.drive_velocity.linear.x
        self.angular_vel = msg.drive_velocity.angular.z

    def odomCallback(self, msg):
        self.linear_vel_feedback = msg.twist.twist.linear.x * 3 / 16
        self.angular_vel_feedback = msg.twist.twist.angular.z

    def joystickCmdCallback(self, msg):
        self.enable_collect_data_joy = msg.enable_collect_data
        self.dir_high_cmd = msg.direct_high_cmd

    def dirHighCmdCallback(self, data):
        self.dir_high_cmd = data.high_cmd.data

    def imuCallback(self, msg):
        self.yaw = np.rad2deg(euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)[2])
        self.imu_msg = msg

    def gnssCallback(self, msg):
        self.gps_msg = msg
        self.gps_error = (msg.position_covariance[0] + msg.position_covariance[4]) * 0.5

    def imageCallback(self, msg):
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgra8")
            self.img_callback_flag = True
            self.secs = msg.header.stamp.secs
            self.nsecs = msg.header.stamp.nsecs
            self.crr_time = self.secs + self.nsecs
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def safetySensorCallback(self, msg):
        self.sonar_right = msg.data_sensor[0]
        self.lidar_front_right = msg.data_sensor[1]
        self.sonar_front_center = msg.data_sensor[2]
        self.lidar_front_center = msg.data_sensor[3]
        self.lidar_front_left = msg.data_sensor[4]
        self.sonar_left = msg.data_sensor[5]
        self.sonar_rear_center = msg.data_sensor[6]

    def saveFrames(self):
        rospy.loginfo(f"Start extracting data to {os.path.join(self.logDir, self.CSV_filename)}")
        with open(os.path.join(self.logDir, self.CSV_filename), 'w') as logFile:
            writer = csv.writer(logFile, delimiter=',', lineterminator='\n')
            writer.writerow([
                "front_center", "angular_velocity(rad/s)", "linear_velocity(m/s)", "dir_high_cmd", 
                "lat", "long", "gps_err", "mag", "imu_orient_x", "imu_orient_y", "imu_orient_z", 
                "imu_orient_w", "imu_ang_vel_x(rad/s)", "imu_ang_vel_y(rad/s)", "imu_ang_vel_z(rad/s)", 
                "imu_lin_acc_x(m/s^2)", "imu_lin_acc_y(m/s^2)", "imu_lin_acc_z(m/s^2)", 
                "sonar_right(m)", "lidar_front_right(m)", "sonar_front_center(m)", "lidar_front_center(m)", 
                "lidar_front_left(m)", "sonar_left(m)", "sonar_rear_center(m)", "angle_to_path", 
                "path", "distance_to_path", "width_path", "full_route"
            ])
            
            while not rospy.is_shutdown():
                try:
                    if self.img_callback_flag:
                        self.img_callback_flag = False
                        if self.old_time == self.crr_time:
                            self.counting += 1
                            rospy.loginfo(f"We have {self.counting} duplicate timestamps!")
                            continue

                        if self.lin_vel_limit[0] < self.linear_vel < self.lin_vel_limit[1]:
                            now = datetime.datetime.fromtimestamp(self.secs).strftime("%Y_%m_%d_%H_%M_%S_")
                            ms = '{:03d}'.format(int(self.nsecs / 1e6))
                            fullNameImage = f"{now}{ms}.jpg"
                            fullpathImage = os.path.join(self.imgDir, fullNameImage)

                            cv2.imwrite(fullpathImage, self.cv_image)

                            writer.writerow([
                                fullNameImage, self.angular_vel_feedback, self.linear_vel_feedback, self.dir_high_cmd,
                                self.gps_msg.latitude, self.gps_msg.longitude, self.gps_error, self.yaw, 
                                self.imu_msg.orientation.x, self.imu_msg.orientation.y, self.imu_msg.orientation.z, 
                                self.imu_msg.orientation.w, self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y, 
                                self.imu_msg.angular_velocity.z, self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y, 
                                self.imu_msg.linear_acceleration.z, self.sonar_right, self.lidar_front_right, 
                                self.sonar_front_center, self.lidar_front_center, self.lidar_front_left, self.sonar_left, 
                                self.sonar_rear_center, self.global_path_nearest.angle_to_path, self.global_path_nearest.array_path, 
                                self.global_path_nearest.distance_to_path, self.global_path_nearest.width_path, self.route
                            ])

                            self.old_time = self.crr_time
                            self.counting = 0
                except Exception as e:
                    rospy.logerr(f"Error: {e}")

    def main(self):
        self.saveFrames()


if __name__ == '__main__':
    a = LogData('front', '/zed2i/zed_node/rgb/image_rect_color/compressed', '/share-work/extracted_data/simulation')
    a.main()
