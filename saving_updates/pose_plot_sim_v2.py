import csv
import utm
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import json
import time
from tqdm import tqdm
import argparse
import re
from auto_data_checking import discontinuity_detector_pro

total_sample = 0


def convert_lat_long_to_utm(lat, lon):
    """Convert latitude and longitude to UTM coordinates."""
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[0], utm_coords[1]


def euler_from_quaternion(x, y, z, w):
    """Convert a quaternion into Euler angles (roll, pitch, yaw)."""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch_y = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def extract_data_and_convert_to_useful_info(file_path, silent=False):
    """Extract data from a CSV file and convert it to useful information."""
    extracted_data = []
    total_error_gps = 0
    num_rows = 0

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            num_rows += 1
            total_error_gps += float(row['gps_err'])

            lat, lon = float(row['lat']), float(row['long'])
            x, y = convert_lat_long_to_utm(lat, lon)

            imu_x, imu_y, imu_z, imu_w = map(float, [row['imu_orient_x'], row['imu_orient_y'], row['imu_orient_z'], row['imu_orient_w']])
            lin_velocity = float(row['linear_velocity(m/s)'])
            angular_velocity = float(row['imu_ang_vel_z(rad/s)'])
            time = row['time']

            roll, pitch, yaw = euler_from_quaternion(imu_x, imu_y, imu_z, imu_w)

            extracted_row = {
                'front_camera': row['front_camera'],
                'x': x, 'y': y,
                'yaw': yaw, 'roll': roll, 'pitch': pitch,
                'linear_velocity': lin_velocity,
                'angular_velocity': angular_velocity,
                'time': time,
            }

            # Optional: Fusion data
            fusion_keys = ['pose_pos_x', 'pose_pos_y', 'pose_pos_z', 'pose_ori_x', 'pose_ori_y', 'pose_ori_z', 'pose_ori_w']
            if all(key in row for key in fusion_keys):
                _, _, fusion_yaw = euler_from_quaternion(
                    float(row['pose_ori_x']),
                    float(row['pose_ori_y']),
                    float(row['pose_ori_z']),
                    float(row['pose_ori_w'])
                )
                extracted_row.update({
                    'pose_pos_x': float(row['pose_pos_x']),
                    'pose_pos_y': float(row['pose_pos_y']),
                    'pose_pos_z': float(row['pose_pos_z']),
                    'fusion_yaw': fusion_yaw,
                })

            extracted_data.append(extracted_row)

    global total_sample
    total_sample += num_rows

    if num_rows > 0 and not silent:
        print(f"Rows processed: {num_rows}")
        print(f"Average GPS error: {total_error_gps / num_rows:.4f}")

    return extracted_data


def transform_coordinates_to_origin_pose(data):
    """Transform coordinates to use the first point as the origin."""
    x_origin, y_origin = data[0]['x'], data[0]['y']
    transformed_x = [row['x'] - x_origin for row in data]
    transformed_y = [row['y'] - y_origin for row in data]
    return transformed_x, transformed_y


def time_str2float(time):
    """Convert time string (yyyy_mm_dd_hh_mm_ss_sss) to seconds."""
    time_parts = list(map(int, time.split('_')[3:]))
    hour, minute, second, millisecond = time_parts[:4]
    return hour * 3600 + minute * 60 + second + millisecond / 1000


def estimate_path_from_yaw(starting_point, data):
    """Estimate the robot's path based on yaw angles, velocity, and time."""
    x, y = starting_point
    path = [[x, y]]

    for i in range(1, len(data)):
        yaw = data[i]['yaw']
        velocity = data[i]['linear_velocity']
        delta_time = time_str2float(data[i]['time']) - time_str2float(data[i - 1]['time'])

        x += velocity * delta_time * np.cos(yaw)
        y += velocity * delta_time * np.sin(yaw)
        path.append([x, y])

    return path


def plot_transformed_coordinates(data):
    """Plot transformed coordinates with estimated paths."""
    transformed_x, transformed_y = transform_coordinates_to_origin_pose(data)
    angular_path = estimate_path_from_yaw((transformed_x[0], transformed_y[0]), data)

    plt.scatter(
        [point[0] for point in angular_path],
        [point[1] for point in angular_path],
        label='Estimated Path',
        s=1,
        color='orange'
    )

    plt.xlabel('Transformed X Coordinate')
    plt.ylabel('Transformed Y Coordinate')
    plt.title('Path Visualization')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


class AutoDetector:
    def __init__(self, detector):
        self.detector = detector

    def auto_check_data(self, data_dir, log_path='autodetected_wrong_data.log'):
        """Automatically check data for discontinuities and log errors."""
        files_to_check = [os.path.join(root, dir, 'final.csv') for root, dirs, _ in os.walk(data_dir) for dir in dirs]
        total_files = len(files_to_check)
        detected_errors = 0

        for file in tqdm(files_to_check, desc='Checking files', total=total_files, unit='files'):
            data = extract_data_and_convert_to_useful_info(file, silent=True)
            if not data:
                continue
            trajectory = np.array(transform_coordinates_to_origin_pose(data)).T
            discontinuities = self.detector.detect_discontinuities(trajectory)

            if np.sum(discontinuities) > 30:
                detected_errors += 1
                WrongDataHandler.append_or_update(file, "Discontinuities detected", log_path)

        return total_files, detected_errors


class WrongDataHandler:
    @staticmethod
    def append_or_update(data_dir, comment, log_path='wrong_data.log', update=False):
        """Append or update a comment in the log file."""
        line = f"{data_dir}\t{comment}\n"
        if update:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            with open(log_path, 'w') as f:
                for line_ in lines:
                    f.write(line if data_dir in line_ else line_)
        else:
            with open(log_path, 'a') as f:
                f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-check data')
    parser.add_argument('--autocheck', action='store_true', help="Apply auto-checking")
    args = parser.parse_args()

    if args.autocheck:
        detector = discontinuity_detector_pro.InterpolationBasedDetector(
            window_size=60,
            threshold=0.005,
            step_size=1,
            order=3
        )
        auto_detector = AutoDetector(detector)
        files_checked, errors_detected = auto_detector.auto_check_data('/path/to/data')
        print(f"{files_checked} files checked, {errors_detected} errors detected.")
