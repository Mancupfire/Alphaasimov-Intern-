#!/usr/bin/env python2

import time
import datetime
import numpy as np
import os
import cv2
import csv
import rospy
import argparse
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, Imu, CompressedImage
from cargobot_msgs.msg import DeliveryState, Safety
from collect_data.msg import GlobalPath, HighCmd
from cv_bridge import CvBridge, CvBridgeError


# Constants
LINEAR_MAX = 3.0  # Maximum linear velocity in m/s
ANGULAR_MAX = 1.0  # Maximum angular velocity in rad/s
LIN_VEL_LIMIT_DEFAULT = [0.02, 2.5]
AP_MODE = 2


# Argument parser
parser = argparse.ArgumentParser(description="Extract data from ROS bag")
parser.add_argument("--bag", dest="bag_file", default="/2020_01_01", type=str, help="Name of the ROS bag file")
parser.add_argument("--rate", dest="rate", default=1, type=int, help="Multiplier for ROS bag rate")
args = parser.parse_args()


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).

    Args:
        x, y, z, w (float): Quaternion components.

    Returns:
        tuple: (roll, pitch, yaw) in radians.
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return roll, pitch, yaw


class LogData:
    def __init__(self, camera_name, topic_name, path_dir):
        """
        Initialize the logging object for data collection.

        Args:
            camera_name (str): Name of the camera.
            topic_name (str): ROS topic for image data.
            path_dir (str): Directory to save the data.
        """
        self.camera_name = camera_name
        self.folder_name = args.bag_file.split('/')[-1][:10]
        self.logDir = os.path.join(path_dir, self.folder_name)
        self.imgDir = os.path.join(self.logDir, 'IMG')
        self.CSV_filename = f"{self.folder_name}.csv"

        # Create directories
        os.makedirs(self.imgDir, exist_ok=True)
        print(f"Created save folder at {self.logDir}")

        # Initialize variables
        self.imu_msg = Imu()
        self.gps_msg = NavSatFix()
        self.cv_image = None
        self.yaw = -1.0
        self.gps_error = -1.0
        self.enable_collect_data_gps = False
        self.img_callback_flag = False
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.lin_vel_limit = LIN_VEL_LIMIT_DEFAULT
        self.global_path_nearest = GlobalPath()
        self.counting = 0
        self.threshold = 30 * args.rate
        self.bridge = CvBridge()
        self.rate = rospy.Rate(args.rate * 30)

        # ROS Subscribers
        rospy.init_node("collect_drive_data_node", anonymous=True)
        rospy.Subscriber("/delivery_state", DeliveryState, self.deliveryStateCallback)
        rospy.Subscriber("/ublox/fix", NavSatFix, self.gnssCallback)
        rospy.Subscriber("/imu/data", Imu, self.imuCallback)
        rospy.Subscriber("/safety_limit_speed", Safety, self.safetySensorCallback)
        rospy.Subscriber("/global_path_nearest", GlobalPath, self.global_path_Callback)
        rospy.Subscriber("/high_cmd", HighCmd, self.highCmdCallback)
        rospy.Subscriber(topic_name, CompressedImage, self.imageCallback)

    def deliveryStateCallback(self, data):
        """
        Callback to update delivery state and GPS collection status.

        Args:
            data (DeliveryState): Delivery state message.
        """
        self.delivery_state = data.delivery_state_monitor
        self.enable_collect_data_gps = self.delivery_state in [4, 5, 7, 8]

    def global_path_Callback(self, data):
        """
        Callback to update global path data.

        Args:
            data (GlobalPath): Global path message.
        """
        self.global_path_nearest = data.global_path

    def imuCallback(self, msg):
        """
        Callback to update IMU data and yaw angle.

        Args:
            msg (Imu): IMU message.
        """
        self.yaw = np.rad2deg(euler_from_quaternion(msg.orientation.x, msg.orientation.y,
                                                    msg.orientation.z, msg.orientation.w)[2])
        self.imu_msg = msg

    def gnssCallback(self, msg):
        """
        Callback to update GPS data.

        Args:
            msg (NavSatFix): GPS message.
        """
        self.gps_msg = msg
        self.gps_error = (msg.position_covariance[0] + msg.position_covariance[4]) * 0.5

    def imageCallback(self, msg):
        """
        Callback to update image data.

        Args:
            msg (CompressedImage): Compressed image message.
        """
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgra8")
            self.img_callback_flag = True
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def safetySensorCallback(self, msg):
        """
        Callback to update safety sensor data.

        Args:
            msg (Safety): Safety sensor message.
        """
        self.sonar_right = msg.data_sensor[0]
        self.lidar_front_center = msg.data_sensor[3]

    def saveFrames(self):
        """
        Save frames and sensor data to files.
        """
        rospy.loginfo(f"Start extracting data to {os.path.join(self.logDir, self.CSV_filename)}")
        with open(os.path.join(self.logDir, self.CSV_filename), "w") as logFile:
            writer = csv.writer(logFile, delimiter=",", lineterminator="\n")
            writer.writerow(["timestamp", "angular_velocity", "linear_velocity", "gps_lat", "gps_long", "gps_error",
                             "yaw", "sonar_right", "lidar_front_center"])

            while not rospy.is_shutdown():
                if self.counting > self.threshold:
                    rospy.loginfo("Data extraction complete. Exiting...")
                    return

                if self.img_callback_flag and self.linear_vel > self.lin_vel_limit[0]:
                    self.img_callback_flag = False
                    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
                    image_name = f"{timestamp}.jpg"
                    image_path = os.path.join(self.imgDir, image_name)

                    # Save image and write data to CSV
                    cv2.imwrite(image_path, self.cv_image)
                    writer.writerow([timestamp, self.angular_vel, self.linear_vel, self.gps_msg.latitude,
                                     self.gps_msg.longitude, self.gps_error, self.yaw, self.sonar_right,
                                     self.lidar_front_center])

                self.rate.sleep()

    def main(self):
        self.saveFrames()


if __name__ == "__main__":
    logger = LogData("front", "/zed2i/zed_node/rgb/image_rect_color/compressed", "/share-work/extracted_data")
    logger.main()
