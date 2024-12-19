#!/usr/bin/env python3.8
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import datetime
import rosbag, sys, csv
import time
import os #for file management make directory
import argparse
from synchronize_v2 import synchronize

SAVE_DIR = '/media/asimovsimpc/bulldog/aa-data/extracted_data/bulldog/umtn/'
BAGFILE = '/media/asimovsimpc/bulldog/temp_data/2023_11_24/compressd/2023_11_24_11_46_24.bag'

parser = argparse.ArgumentParser(description='Extract data from rosbag')
parser.add_argument('--bag', dest='bag_file', default=BAGFILE, type=str, help='Name of the rosbag file')
parser.add_argument('--dir', dest='extract_dir', default=SAVE_DIR, type=str, help='Extracted directory.')
parser.add_argument('--rate', dest='rate', default=25, type=int, help='Rate of saving data (Hz)')
parser.add_argument('--data', dest='data_type', default='umtn', type=str, help='Data type: umtn or bc')
parser.add_argument('--scenario', dest='scenario', default='mix', type=str, help='Scenario, can be 1.1, 1.2,... or mix')
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


def write_imu(writer, msg, date_time):
    yaw = np.rad2deg(euler_from_quaternion(msg.orientation.x,
                                           msg.orientation.y,
                                           msg.orientation.z,
                                           msg.orientation.w)[2])
    
    sensor_data = [date_time, yaw,
                   msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                   msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                   msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

    writer.writerow(sensor_data)


def write_high_cmd( writer, msg, date_time):
    dir_high_cmd = msg.remote_control.direct_high_cmd
    sensor_data = [date_time, dir_high_cmd]

    writer.writerow(sensor_data)


def write_speed(writer, msg, date_time):
    linear_vel = msg.twist.twist.linear.x
    angular_vel = msg.twist.twist.angular.z
    
    sensor_data = [date_time, angular_vel, linear_vel]

    writer.writerow(sensor_data)


def write_gps(writer, msg, date_time):
    gps_error = (msg.position_covariance[0] + msg.position_covariance[4]) * 0.5

    sensor_data = [date_time, msg.latitude, msg.longitude, gps_error]

    writer.writerow(sensor_data)


def write_path(writer, msg, date_time):
    global_path_nearest = msg.global_path
    # global_path_nearest = msg

    sensor_data = [date_time,
                   global_path_nearest.distance_to_path,
                   global_path_nearest.array_path,
                   global_path_nearest.width_path]

    writer.writerow(sensor_data)


def bag2csv(bag, logDir, dataType):
	#get list of topics from the bag
    if 'umtn' in dataType:
        listOfTopics = ["/imu/data", "/remote_control", "/odom", "/ublox/fix", "/global_path_nearest"]
        csv_filename = ["imu", "high_cmd", "speed", "gps", "global_path"]
        function_dict = {
            "/imu/data": write_imu,
            "/remote_control": write_high_cmd,
            "/odom": write_speed,
            "/ublox/fix": write_gps,
            "/global_path_nearest": write_path,
        }

        data_info = [
            ["mag", "imu_orient_x", "imu_orient_y", "imu_orient_z", "imu_orient_w",
            "imu_ang_vel_x(rad/s)", "imu_ang_vel_y(rad/s)", "imu_ang_vel_z(rad/s)",
            "imu_lin_acc_x(m/s^2)", "imu_lin_acc_y(m/s^2)", "imu_lin_acc_z(m/s^2)"],
            ["dir_high_cmd"],
            ["angular_velocity(rad/s)", "linear_velocity(m/s)"],
            ["lat", "long", "gps_err"],
            ["angle_to_path","path","distance_to_path","width_path"],
        ]
    else: # BC data
        listOfTopics = ["/imu/data", "/remote_control", "/odom", "/ublox/fix"]
        csv_filename = ["imu", "high_cmd", "speed", "gps"]
        function_dict = {
            "/imu/data": write_imu,
            "/remote_control": write_high_cmd,
            "/odom": write_speed,
            "/ublox/fix": write_gps,
        }

        data_info = [
            ["mag", "imu_orient_x", "imu_orient_y", "imu_orient_z", "imu_orient_w",
            "imu_ang_vel_x(rad/s)", "imu_ang_vel_y(rad/s)", "imu_ang_vel_z(rad/s)",
            "imu_lin_acc_x(m/s^2)", "imu_lin_acc_y(m/s^2)", "imu_lin_acc_z(m/s^2)"],
            ["dir_high_cmd"],
            ["angular_velocity(rad/s)", "linear_velocity(m/s)"],
            ["lat", "long", "gps_err"],
        ]

    filename = ""
    for id, topicName in enumerate(listOfTopics):
        #Create a new CSV file for each topic
        filename = logDir + '/' + csv_filename[id] + '.csv'
        old_time = 0

        with open(filename, 'w+') as csvfile:
            writer=csv.writer(csvfile, delimiter=',', lineterminator='\n',)
            writer.writerow(["time"] + data_info[id])

            extracted_data = bag.read_messages(topicName)
            for seq_id, bag_data in enumerate(extracted_data):  
                msg = bag_data.message     
                secs = msg.header.stamp.secs 
                nsecs = msg.header.stamp.nsecs

                if old_time == (secs + nsecs*1e-9):
                    continue
                else:
                    old_time = secs + nsecs*1e-9
                    ms = '{:03d}'.format((int)(nsecs/1e6))
                    date_time = datetime.datetime.fromtimestamp(secs).strftime("%Y_%m_%d_%H_%M_%S_") + ms
                    function_dict[topicName](writer, msg, date_time)
        
        print(f"Finish saving {topicName} to {csv_filename[id]}")

    print("Finish saving all data to csv files")


def create_log_folder(bag_path):
    if not os.path.exists(args.extract_dir):
        os.makedirs(args.extract_dir)
        print("Create save folder at {}".format(args.extract_dir))

    # create/get location/area where the data was collected folder
    folder_name = bag_path.split('/')[-4]
    logDir = os.path.join(args.extract_dir, folder_name)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        print("Create save folder at {}".format(logDir))
    
    # create/get scenario folder
    folder_name = bag_path.split('/')[-3]
    logDir = os.path.join(logDir, folder_name)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        print("Create save folder at {}".format(logDir))

    # create/get date folder
    folder_name = bag_path.split('/')[-2][:10]
    logDir = os.path.join(logDir, folder_name)
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        print("Create save folder at {}".format(logDir))

    # create/get time folder
    folder_name = bag_path.split('/')[-1][:-4]
    logDir = os.path.join(logDir, folder_name)    
    if not os.path.exists(logDir):
        os.makedirs(logDir)
        print("Create save sub-folder at {}".format(logDir))

    return logDir


def bag2png(bag, logDir, dataType):
    if dataType == 'umtn':
        list_of_topics = ["/camera_front", "/camera_left", "/camera_right",
                        "/camera_back", "/camera_traffic_light", "/routed_map/compressed"]
    elif dataType == 's-umtn':
        list_of_topics = ["/camera_front", "/routed_map/compressed"]
    elif dataType == 'bc':
        list_of_topics = ["/camera_front", "/camera_left", "/camera_right",
                        "/camera_back", "/camera_traffic_light"]
    else:
        list_of_topics = ["/camera_front"]
    # list_of_topics = ["/camera_front", "/routed_map/compressed"]

    bridge = CvBridge()
    
    sec_list = []
    nsec_list = []

    for topic in list_of_topics:
        print(f"Reading topic {topic} ...")
        topic_name = topic.split('/')[1]
        imgDir = os.path.join(logDir, topic_name)
        if not os.path.exists(imgDir):
            os.mkdir(imgDir)

        image_topic = bag.read_messages(topic)
        for id, b in enumerate(image_topic):           
            message = b.message 
            cv_image = bridge.compressed_imgmsg_to_cv2(message)

            secs = message.header.stamp.secs
            nsecs = message.header.stamp.nsecs
            
            if (topic == '/camera_front'):
                if secs == 0 and nsecs == 0:
                    continue

                if (len(sec_list) == 0) or (sec_list[-1] != secs) or (nsec_list[-1] != nsecs):
                    sec_list.append(secs)
                    nsec_list.append(nsecs)
                else:
                    continue

            if (topic == "/routed_map/compressed"):
                if id >= len(sec_list):
                    crr_id = len(sec_list) - 1
                else:
                    crr_id = id

                secs = sec_list[crr_id]
                nsecs = nsec_list[crr_id]

            now = datetime.datetime.fromtimestamp(secs).strftime("_%Y_%m_%d_%H_%M_%S_")
            ms = '{:03d}'.format((int)(nsecs/1e6))    
            fullNameImage = topic_name + now + ms + '.jpg'

            # fullNameImage = str(id) + '.jpg'
            # cv_image.astype(np.uint8)
            cv2.imwrite(os.path.join(imgDir, fullNameImage), cv_image)
        
        print(f'Finish saving topic {topic}!')
    
    return logDir


if __name__ == '__main__':
    bag = rosbag.Bag(args.bag_file)    

    logDir = create_log_folder(args.bag_file)
    bag2png(bag, logDir, args.data_type)
    bag2csv(bag, logDir, args.data_type)
    synchronize(logDir, logDir, args.rate, args.data_type, args.scenario)

    bag.close()
    print('PROCESS COMPLETE')