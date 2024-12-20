import csv
import utm
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import cv2
import time
from tqdm import tqdm
import argparse
import re
from auto_data_checking import discontinuity_detector_fit_point

# Global Variables
total_sample = 0

# Utility Functions
def convert_lat_long_to_utm(lat, lon):
    """Convert latitude and longitude to UTM coordinates."""
    utm_coords = utm.from_latlon(lat, lon)
    return utm_coords[0], utm_coords[1]


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def extract_data_and_convert_to_useful_info(file_path, silent=False):
    """Extract data from CSV and convert it into a usable format."""
    extracted_data = []
    i = 0
    error_gps = 0

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            i += 1
            error_gps += float(row['gps_err'])
            lat = float(row['lat'])
            lon = float(row['long'])
            x, y = convert_lat_long_to_utm(lat, lon)

            imu_orient = [float(row[f'imu_orient_{axis}']) for axis in ['x', 'y', 'z', 'w']]
            lin_velocity = float(row['linear_velocity(m/s)'])
            angular_velocity = float(row['imu_ang_vel_z(rad/s)'])
            time = row['time']

            roll, pitch, yaw = euler_from_quaternion(*imu_orient)

            extracted_row = {
                'front_camera': row['front_camera'],
                'x': x, 'y': y,
                'yaw': yaw, 'roll': roll, 'pitch': pitch,
                'linear_velocity': lin_velocity,
                'angular_velocity': angular_velocity,
                'time': time,
            }

            fusion_keys = ['pose_pos_x', 'pose_pos_y', 'pose_pos_z', 'pose_ori_x', 'pose_ori_y', 'pose_ori_z', 'pose_ori_w']
            if all(key in row for key in fusion_keys):
                fusion_orient = [float(row[f'pose_ori_{axis}']) for axis in ['x', 'y', 'z', 'w']]
                _, _, fusion_yaw = euler_from_quaternion(*fusion_orient)
                extracted_row.update({
                    'pose_pos_x': float(row['pose_pos_x']),
                    'pose_pos_y': float(row['pose_pos_y']),
                    'pose_pos_z': float(row['pose_pos_z']),
                    'pose_ori_x': fusion_orient[0],
                    'pose_ori_y': fusion_orient[1],
                    'pose_ori_z': fusion_orient[2],
                    'pose_ori_w': fusion_orient[3],
                    'fusion_yaw': fusion_yaw,
                })

            extracted_data.append(extracted_row)

    global total_sample
    total_sample += i

    if not silent:
        print(f"Processed {i} rows. GPS error: {(error_gps + 0.0001) / i}")

    return extracted_data


def transform_coordinates_to_origin(data):
    """Transform coordinates to origin-relative values."""
    x_coords = [row['x'] for row in data]
    y_coords = [row['y'] for row in data]
    x_origin, y_origin = x_coords[0], y_coords[0]

    transformed_x = [x - x_origin for x in x_coords]
    transformed_y = [y - y_origin for y in y_coords]
    return transformed_x, transformed_y


def time_str_to_seconds(time_str):
    """Convert a timestamp string to seconds."""
    time_parts = list(map(int, time_str.split('_')[3:]))
    hour, minute, second, millisecond = time_parts[0], time_parts[1], time_parts[2], time_parts[3]
    return hour * 3600 + minute * 60 + second + millisecond / 1000


def estimate_path(starting_point, data, use_yaw=True):
    """Estimate the path based on yaw angles or angular velocities."""
    x, y = starting_point
    path = [list(starting_point)]

    for i in range(1, len(data)):
        velocity = data[i]['linear_velocity']
        time_delta = time_str_to_seconds(data[i]['time']) - time_str_to_seconds(data[i - 1]['time'])
        if use_yaw:
            yaw = data[i]['yaw']
            x += velocity * time_delta * np.cos(yaw)
            y += velocity * time_delta * np.sin(yaw)
        else:
            angular_velocity = data[i]['angular_velocity']
            x += velocity * np.cos(angular_velocity * time_delta)
            y += velocity * np.sin(angular_velocity * time_delta)
        path.append([x, y])
    return path


# Class Definitions
class AutoDetector:
    """Class for automatic detection of data anomalies."""

    def __init__(self, detector):
        self.detector = detector

    def auto_check_data(self, data_dir, log_path='autodetected_wrong_data.log'):
        """Automatically check data and log detected anomalies."""
        files_to_check = [
            os.path.join(root, dir, 'final.csv')
            for root, dirs, _ in os.walk(data_dir) for dir in sorted(dirs) if any(year in dir for year in ['2023', '2024'])
        ]

        numb_files = len(files_to_check)
        numb_detected = 0

        for csv_path in tqdm(files_to_check, desc="Checking files", unit="files"):
            data = extract_data_and_convert_to_useful_info(csv_path, silent=True)
            if not data:
                continue

            data_array = np.array(transform_coordinates_to_origin(data)).T
            discontinuities = self.detector.detect_discontinuities(data_array)
            if np.sum(discontinuities) < 30:
                continue

            WrongDataHandler.append_or_update(data_dir=os.path.dirname(csv_path), comment="Discontinuity detected")
            numb_detected += 1

        return numb_files, numb_detected


class WrongDataHandler:
    """Class for handling wrong data and logging."""

    @staticmethod
    def append_or_update(data_dir, comment, log_path='wrong_data.log', update=False):
        """Append or update log file with wrong data details."""
        line = f"{data_dir}\t{comment}\n"
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(line)
        else:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            with open(log_path, 'w') as f:
                for l in lines:
                    if data_dir in l:
                        f.write(line)
                    else:
                        f.write(l)
                if not update:
                    f.write(line)


# Main Script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto-check data for discontinuities.")
    parser.add_argument('--autocheck', action='store_true', help="Perform autocheck on the data.")
    args = parser.parse_args()

    with open('pose_plot_config.json') as f:
        config = json.load(f)

    detector = discontinuity_detector_fit_point.InterpolationBasedDetector(window_size=60, threshold=0.01, step_size=2, order=4)

    if args.autocheck:
        data_path = config['path']
        auto_detector = AutoDetector(detector)
        total_files, total_detected = auto_detector.auto_check_data(data_path)
        print(f"Checked {total_files} files. Detected {total_detected} issues.")
