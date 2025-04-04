#!/usr/bin/env python3.8

###!/home/alaska02/trand/bin/python3
import time
import datetime
import numpy as np
import os
import cv2
import csv
import rospy
import copy

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, Joy, CompressedImage
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError

from cargobot_msgs.msg import RemoteControl, JoystickControl
from cargobot_msgs.msg import DriveState, DeliveryState
from cargobot_msgs.msg import Safety, Route
from collect_data.msg import GlobalPath, RemoteControl, HighCmd
from scout_msgs.msg import ScoutStatus
# from tf.transformations import euler_from_quaternion
import argparse

LINEAR_MAX = 3.0 # m/s
ANGULAR_MAX = 1.0 # rad/s
LIN_VEL_LIMIT_DEFAULT = [-10, 14]
# LIN_VEL_LIMIT_DEFAULT = [0.02, 3.5]
STEP = 0.1
AP_MODE = 2

parser = argparse.ArgumentParser(description='Extract data from rosbag')
parser.add_argument('-b', '--bag', dest='bag_file', default='/media/asimovsimpc/bulldog/aa-data/bulldog/compress/phenikaa/2023_11_21_b01/2023_11_21_16_38_03.bag', type=str, help='Name of the rosbag file')
#parser.add_argument('--rate', dest='rate', default=1, type=int, help='multiply rosbag rate record')
#parser.add_argument('--fps', dest='fps', default=20, type=int, help='frame rate of rosbag record')
args = parser.parse_args()


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians


class LogData():
    def __init__(self, camera_name, topic_name, path_dir):
        self.path_dir_img = []
        print(camera_name)
        self.sensor_list = camera_name
        # create/get location folder
        self.folder_name = args.bag_file.split('/')[-4]
        self.logDir = os.path.join( path_dir, self.folder_name)
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
            print("Create save folder at {}".format(self.logDir))
        
        # create/get scenario folder
        self.folder_name = args.bag_file.split('/')[-3]
        self.logDir = os.path.join( self.logDir, self.folder_name)
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
            print("Create save folder at {}".format(self.logDir))

        # create/get date folder
        self.folder_name = args.bag_file.split('/')[-2][:10]
        self.logDir = os.path.join( self.logDir, self.folder_name)
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
            print("Create save folder at {}".format(self.logDir))

        # create/get time folder
        self.folder_name = args.bag_file.split('/')[-1][:-4]

        self.CSV_filename = self.folder_name + '.csv'
        self.log_filename = self.folder_name + '_statistic.csv'

        self.logDir = os.path.join( self.logDir, self.folder_name)
        # self.imgDir = os.path.join( self.logDir,'IMG')

        for name in camera_name:
            self.folder_name = str(name)
            temp_logDir = os.path.join( self.logDir, self.folder_name)            
            self.path_dir_img.append(temp_logDir)
            if not os.path.exists(temp_logDir):
                os.makedirs(temp_logDir)
        
        print(type(self.sensor_list[0]))

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

        self.sensor_images = dict()
        self.crr_time = dict()
        self.image_names = dict()

        # self.cv_image_front = None
        # self.cv_image_left = None
        # self.cv_image_right = None
        # self.cv_image_back = None
        # self.cv_image_traffic_light = None

        self.img_callback_flag = False
        self.enable_collect_data_rc = False
        self.enable_collect_data_joy = False
        self.X_axes = 0
        self.Y_axes = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.linear_vel_feedback = 0.0
        self.angular_vel_feedback = 0.0
        self.dir_high_cmd_rc = 0.0
        self.dir_high_cmd_joy = 0.0
        self.dir_high_cmd = 0.0
        self.mode = 0
        
        self.secs = 0
        self.nsecs = 0
        
        self.counting = 0
        self.crr_seq = 0
        self.old_seq = 0


        rospy.init_node('collect_drive_data_node', anonymous = True) 
        rospy.Subscriber('/delivery_state', DeliveryState, self.deliveryStateCallback)
        rospy.Subscriber("/ublox/fix", NavSatFix, self.gnssCallback)
        rospy.Subscriber("/imu/data", Imu, self.imuCallback)
        
        rospy.Subscriber("/safety_limit_speed", Safety, self.safetySensorCallback)
        
        rospy.Subscriber("/high_cmd", HighCmd, self.dirHighCmdCallback)

        # rospy.Subscriber("/remote_control", RemoteControl, self.remoteCmdCallback)
        rospy.Subscriber("/scout_status", ScoutStatus, self.scoutStatusCmdCallback)
        rospy.Subscriber("/joystick_control", JoystickControl, self.joystickCmdCallback)
        rospy.Subscriber("/drive_state", DriveState, self.driveStateCallback)
        rospy.Subscriber("/global_path_nearest", GlobalPath, self.global_path_Callback)
        rospy.Subscriber("/route", Route, self.route_Callback)
        # rospy.Subscriber("/odom", Odometry, self.odomCallback)

        # rospy.Subscriber(topic_name, Image, self.imageCallback)
        rospy.Subscriber("/camera_front", CompressedImage, self.imageCamFrontCallback)
        rospy.Subscriber('/camera_left', CompressedImage, self.imageCamLeftCallback)
        rospy.Subscriber('/camera_right', CompressedImage, self.imageCamRightCallback)
        rospy.Subscriber("/camera_back", CompressedImage, self.imageCamBackCallback)
        rospy.Subscriber("/camera_traffic_light", CompressedImage, self.imageCamTFLCallback)
        

        self.bridge = CvBridge()
        # self.rate = rospy.Rate(args.rate * args.fps)
        # # self.rate = rospy.Rate(30)
        # self.threash_rate = args.fps * args.rate

        self.lin_vel_limit = LIN_VEL_LIMIT_DEFAULT
        self.delivery_state = 0
        self.enable_collect_data_gps = False

        self.global_path_nearest = GlobalPath()
        self.route = []
        # self.remote_control = RemoteControl()
        # self.scout_status_msg = ScoutStatus()
        self.imu_msg = Imu()
        self.gps_msg = NavSatFix()
        self.high_cmd = HighCmd()        
        
    def deliveryStateCallback(self, data):
        """
        Delivery state:
          0: Undefined state
          1: Order Waiting
          2: Order Received
          3: Route Available
          4: Store Coming
          5: Store Arrived
          6: Package Sent
          7: Customer Coming
          8: Customer Arrived
          9: Customer Encountered
          10: Package Delivered
          101: Route Unavailable
        
        """
        self.delivery_state = data.delivery_state_monitor
        if self.delivery_state in [4,5,7,8] :
            self.enable_collect_data_gps = True
        else:
            self.enable_collect_data_gps = False


    def global_path_Callback(self,data):
        self.global_path_nearest = data.global_path

    def route_Callback(self,data):
        self.route = data.list_point
    
    def driveStateCallback(self, msg):
        self.drive_state = msg.drive_mode_state
        # self.linear_vel = msg.drive_velocity.linear.x
        # self.angular_vel = msg.drive_velocity.angular.z

    def odomCallback(self, msg):
        self.linear_vel_feedback = msg.twist.twist.linear.x
        self.angular_vel_feedback = msg.twist.twist.angular.z

    def joystickCmdCallback(self, msg):
        # ROS Subscriber remote_control topic callback function:
        self.enable_collect_data_joy = msg.enable_collect_data
        self.dir_high_cmd_joy = msg.direct_high_cmd


    # def remoteCmdCallback(self, data):
        # ROS Subscriber remote_control topic callback function:
        # self.enable_collect_data_rc = data.enable_collect_data
        # if data.operation_mode_cmd==1:
        #     self.dir_high_cmd_rc = 0
        # elif data.operation_mode_cmd==0:
        #     self.dir_high_cmd_rc = -1
        # elif data.operation_mode_cmd==2:
        #     self.dir_high_cmd_rc = 1
        # self.dir_high_cmd_rc = msg.direct_high_cmd
        # self.linear_vel = data.remote_control.remote_vel_cmd.linear.x
        # self.angular_vel = data.remote_control.remote_vel_cmd.angular.z

    def scoutStatusCmdCallback(self, data):
        self.linear_vel = data.linear_velocity
        self.angular_vel = data.angular_velocity

       
        # self.img_callback_flag = True
        # self.secs = data.header.stamp.secs
        # self.nsecs = data.header.stamp.nsecs
        # self.crr_time = self.secs + self.nsecs

    def dirHighCmdCallback(self, data):
        self.dir_high_cmd = data

    def imuCallback(self, msg):
        """
            Topic "/imu/mag" return orientation in vector with:
                x: to the North
                y: to the East
            Conversion is needed to transform magnetic-orientation to "earth coordinate"
            Return angle from -180 to 180 degree 
        """
        self.yaw = np.rad2deg(euler_from_quaternion(msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w)[2])
        self.imu_msg = msg

        # self.secs = msg.header.stamp.secs
        # self.nsecs = msg.header.stamp.nsecs
        # self.crr_time = self.secs + self.nsecs

    def gnssCallback(self, msg):
        """ 
        Receive GNSS data from ublox_node
        """
        # NavSatFix
        self.gps_msg = msg
        self.gps_error = (msg.position_covariance[0] + msg.position_covariance[4]) * 0.5


    def imageCamFrontCallback(self, msg):
        # Image topic callback
        try:
            # self.cv_image_front = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.img_callback_flag = True
            
            # self.secs = msg.header.stamp.secs
            # self.nsecs = msg.header.stamp.nsecs
            # self.crr_time = self.secs + self.nsecs
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)

            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            crr_time = secs + nsecs

            # self.sensor_images[self.sensor_list[0]] = cv_image
            self.crr_time[self.sensor_list[0]] = crr_time
        
            now = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))            
            fullNameImage = now + ms + '.jpg'
            self.image_names[self.sensor_list[0]] = fullNameImage

            self.crr_seq = msg.header.seq
            # rospy.loginfo(f"Image shape of {self.sensor_list[0]} camera = {cv_image.shape}")
            
            fullpath = os.path.join(self.path_dir_img[0], fullNameImage)
            cv2.imwrite(fullpath, cv_image)            

        except CvBridgeError as e:
            rospy.logerr("Waiting msg from {}".format(self.sensor_list[0]))
            rospy.logerr("CvBridge Error: {0}".format(e))
    
    
    def imageCamLeftCallback(self, msg):
        # Image topic callback
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            crr_time = secs + nsecs

            # self.sensor_images[self.sensor_list[1]] = cv_image
            self.crr_time[self.sensor_list[1]] = crr_time

            now = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))            
            fullNameImage = now + ms + '.jpg'
            self.image_names[self.sensor_list[1]] = fullNameImage
            # rospy.loginfo(f"Image shape of {self.sensor_list[1]} camera = {cv_image.shape}")
            fullpath = os.path.join(self.path_dir_img[1], fullNameImage)
            cv2.imwrite(fullpath, cv_image)            

        except CvBridgeError as e:
            rospy.logerr("Waiting msg from {}".format(self.sensor_list[1]))
            # rospy.logerr("CvBridge Error: {0}".format(e))


    def imageCamRightCallback(self, msg):
        # Image topic callback
        try:
            # self.cv_image_right = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.img_callback_flag = True

            # self.secs = msg.header.stamp.secs
            # self.nsecs = msg.header.stamp.nsecs
            # self.crr_time = self.secs + self.nsecs

            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            crr_time = secs + nsecs

            # self.sensor_images[self.sensor_list[2]] = cv_image
            self.crr_time[self.sensor_list[2]] = crr_time

            now = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))            
            fullNameImage = now + ms + '.jpg'
            self.image_names[self.sensor_list[2]] = fullNameImage
            # rospy.loginfo(f"Image shape of {self.sensor_list[2]} camera = {cv_image.shape}")
            
            fullpath = os.path.join(self.path_dir_img[2], fullNameImage)
            cv2.imwrite(fullpath, cv_image)            

        except CvBridgeError as e:
            rospy.logerr("Waiting msg from {}".format(self.sensor_list[2]))
            # rospy.logerr("CvBridge Error: {0}".format(e))


    def imageCamBackCallback(self, msg):
        # Image topic callback
        try:
            # self.cv_image_back = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.img_callback_flag = True

            # self.secs = msg.header.stamp.secs
            # self.nsecs = msg.header.stamp.nsecs
            # self.crr_time = self.secs + self.nsecs
            
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            crr_time = secs + nsecs

            # self.sensor_images[self.sensor_list[3]] = cv_image
            self.crr_time[self.sensor_list[3]] = crr_time

            now = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))            
            fullNameImage = now + ms + '.jpg'
            self.image_names[self.sensor_list[3]] = fullNameImage
            # rospy.loginfo(f"Image shape of {self.sensor_list[3]} camera = {cv_image.shape}")
            
            fullpath = os.path.join(self.path_dir_img[3], fullNameImage)
            cv2.imwrite(fullpath, cv_image)            

        except CvBridgeError as e:
            rospy.logerr("Waiting msg from {}".format(self.sensor_list[3]))
            # rospy.logerr("CvBridge Error: {0}".format(e))


    def imageCamTFLCallback(self, msg):
        # Image topic callback
        try:
            # self.cv_image_traffic_light = self.bridge.compressed_imgmsg_to_cv2(msg)
            # self.img_callback_flag = True

            # self.secs = msg.header.stamp.secs
            # self.nsecs = msg.header.stamp.nsecs
            # self.crr_time = self.secs + self.nsecs

            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            

            secs = msg.header.stamp.secs
            nsecs = msg.header.stamp.nsecs
            crr_time = secs + nsecs

            # self.sensor_images[self.sensor_list[4]] = cv_image
            self.crr_time[self.sensor_list[4]] = crr_time

            now = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))            
            fullNameImage = now + ms + '.jpg'
            self.image_names[self.sensor_list[4]] = fullNameImage
            # rospy.loginfo(f"Image shape of {self.sensor_list[4]} camera = {cv_image.shape}")

            fullpath = os.path.join(self.path_dir_img[4], fullNameImage)
            cv2.imwrite(fullpath, cv_image)        

        except CvBridgeError as e:
            rospy.logerr("Waiting msg from {}".format(self.sensor_list[4]))
            # rospy.logerr("CvBridge Error: {0}".format(e))
        

    def safetySensorCallback(self, msg):
        self.sonar_right = msg.data_sensor[0]
        self.lidar_front_right = msg.data_sensor[1]
        self.sonar_front_center = msg.data_sensor[2]
        self.lidar_front_center = msg.data_sensor[3]
        self.lidar_front_left = msg.data_sensor[4]
        self.sonar_left = msg.data_sensor[5]
        self.sonar_rear_center = msg.data_sensor[6]

    def saveFrames(self):
        rospy.loginfo("Start extract data to {}".format(os.path.join(self.logDir, self.CSV_filename)))
        with open(os.path.join(self.logDir, self.CSV_filename), 'w') as logFile:
            writer=csv.writer(logFile, delimiter=',', lineterminator='\n',)
            # writer.writerow(["front_center", "angular_velocity(rad/s)", "linear_velocity(m/s)", "dir_high_cmd", "lat", "long", "gps_err", "mag",
            #                  "imu_orient_x", "imu_orient_y", "imu_orient_z", "imu_orient_w",
            #                  "imu_ang_vel_x(rad/s)", "imu_ang_vel_y(rad/s)", "imu_ang_vel_z(rad/s)", "imu_lin_acc_x(m/s^2)", "imu_lin_acc_y(m/s^2)", "imu_lin_acc_z(m/s^2)",
            #                  "sonar_right(m)", "lidar_front_right(m)", "sonar_front_center(m)", "lidar_front_center(m)", "lidar_front_left(m)", "sonar_left(m)",
            #                  "sonar_rear_center(m)","angle_to_path","path","distance_to_path","width_path","full_route"])
            
            writer.writerow(["front_center", "angular_velocity(rad/s)", "linear_velocity(m/s)", "dir_high_cmd", "lat", "long", "gps_err"])
       
            while not rospy.is_shutdown():
                try:
                    # if self.secs == 0:
                    #     rospy.loginfo(self.secs)
                    #     continue
                    # now = datetime.datetime.now()
                    # ms = '{:03d}'.format((int)(now.microsecond/1000))                    

                    
                    # if (self.enable_collect_data_rc or self.enable_collect_data_joy) and self.img_callback_flag and self.drive_state != AP_MODE:
                    # if self.enable_collect_data_rc and self.enable_collect_data_gps and self.gps_error < 5 and (-2.0<self.global_path_nearest.distance_to_path <2+self.global_path_nearest.width_path/2 ):
                    # if self.enable_collect_data_rc:
                    #     self.dir_high_cmd = self.dir_high_cmd_rc
                    # else :
                    #     self.dir_high_cmd = self.dir_high_cmd_joy

                    # rospy.loginfo(self.linear_vel)

                    # if self.counting > self.threash_rate:
                    #     rospy.loginfo("Finish extracting data, exiting...")
                    #     return

                    # if self.img_callback_flag == True:
                    #     # print('debug')
                    #     self.img_callback_flag = False

                    if self.old_seq == self.crr_seq:
                        self.counting += 1
                        #rospy.loginfo("We have {} dupplicate here!".format(self.counting))                            
                        continue
                        
                    #     if self.linear_vel > self.lin_vel_limit[0] and self.linear_vel < self.lin_vel_limit[1]:
                    #         now = datetime.datetime.fromtimestamp(self.secs).strftime("%Y_%m_%d_%H_%M_%S_")
                    #         ms = '{:03d}'.format((int)(self.nsecs/1e6))

                    #         # fullNameImage = time.strftime("%Y_%m_%d_%H_%M_%S_") + ms + '.jpg'
                    #         fullNameImage = now + ms + '.jpg'
                    #         fullpathImageFront = os.path.join(self.path_dir_img[0], fullNameImage)
                    #         fullpathImageLeft = os.path.join(self.path_dir_img[1], fullNameImage)
                    #         fullpathImageRight = os.path.join(self.path_dir_img[2], fullNameImage)
                    #         fullpathImageBack = os.path.join(self.path_dir_img[3], fullNameImage)
                    #         fullpathImageTFL = os.path.join(self.path_dir_img[4], fullNameImage)
                            
                    #         # print(fullpathImage)
                            
                    #         # check empty_img to skip. It happens at the beginning of rosbag play
                    #         # if self.cv_image is None:
                    #         #     continue
                    #         #print(self.cv_image_left)
                    #         # # save image
                    #         cv2.imwrite(fullpathImageFront, self.cv_image_front)
                            #cv2.imwrite(fullpathImageLeft, self.cv_image_left)
                            #cv2.imwrite(fullpathImageRight, self.cv_image_right)
                            #cv2.imwrite(fullpathImageBack, self.cv_image_back)
                            #cv2.imwrite(fullpathImageTFL, self.cv_image_traffic_light)
                            
                            # fix_path = ''' + self.global_path_nearest.array_path + '"'
                            # print(fix_path)
                            # write to file CSV
                    # writer.writerow([self.image_names['front'].split('/')[-1], self.angular_vel, self.linear_vel, self.dir_high_cmd, self.gps_msg.latitude, self.gps_msg.longitude, self.gps_error, self.yaw,
                    #                     self.imu_msg.orientation.x, self.imu_msg.orientation.y, self.imu_msg.orientation.z, self.imu_msg.orientation.w,
                    #                     self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y, self.imu_msg.angular_velocity.z, self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y, self.imu_msg.linear_acceleration.z,
                    #                     self.sonar_right, self.lidar_front_right, self.sonar_front_center, self.lidar_front_center, self.lidar_front_left, self.sonar_left, self.sonar_rear_center,
                    #                     self.global_path_nearest.angle_to_path, self.global_path_nearest.array_path, self.global_path_nearest.distance_to_path, self.global_path_nearest.width_path, self.route])

                    writer.writerow([self.image_names['front'].split('/')[-1], self.angular_vel, self.linear_vel, self.dir_high_cmd, self.gps_msg.latitude, self.gps_msg.longitude, self.gps_error])        
                            # print(self.route)
                    # if self.counting > 100:
                    #     break
                    
                    # rospy.loginfo(f"number of sensor image {len(self.sensor_images)}")

                    # if len(self.sensor_images) == len(self.sensor_list):
                    #     for id, dir_path  in enumerate(self.path_dir_img):
                    #         sensor_name = self.sensor_list[id]
                    #         save_path = os.path.join(dir_path, self.image_names[sensor_name])
                    #         rospy.loginfo(f"Saving data to {save_path}")
                    #         cv2.imwrite(save_path, self.sensor_images[sensor_name])

                    self.old_seq = self.crr_seq
                    self.counting = 0  

                   
                except Exception as E:
                    print("error",E)
    def main(self):
        # self.setLimitVel()
        self.saveFrames()

if __name__ == '__main__':
    # a = LogData('front', '/zed2i/zed_node/rgb/image_rect_color/compressed', '/home/asimovsimpc/share-work/process_data')
    # a = LogData('front', '/image_raw', '/share-work/extracted_data')
    a = LogData(["front","left","right","back","traffic_light"], '/front', '/media/asimovsimpc/bulldog/aa-data/extracted_data/bulldog')
    # a = LogData('front', '/zed2i/zed_node/rgb/image_rect_color/compressed', '/share-work')
    a.main()