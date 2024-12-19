import csv
import utm
import math 
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
import json
import cv2
import matplotlib
import time
from tqdm import tqdm
import argparse
import re
from auto_data_checking import discontinuity_detector_pro


total_sample = 0

def convert_lat_long_to_utm(lat, long):
    utm_coords = utm.from_latlon(lat, long)
    return utm_coords[0], utm_coords[1]

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

def offset(list,width):
    x_l=[]
    y_l=[]
    x_r=[]
    y_r=[]
    x=[]
    y=[]
    points_right= []
    points_left= []
    l = len(list)
    for i in range(l):
        if i==1:
            
            offset =width-0.5
            alpha1 = 0.0
            alpha2 = 0.0
            alpha_r_of = 0.0
            alpha_l_of = 0.0
            l= len(list)
        
            alpha1 = math.atan2(list[i][1] - list[i-1][1], list[i][0] - list[i-1][0])
            alpha2 = math.atan2(list[i+1][1] - list[i][1], list[i+1][0] - list[i][0])
            
            alpha_r_of = (alpha1 + alpha2)/2 + math.pi/2
            alpha_l_of = (alpha1 + alpha2)/2 - math.pi/2
            x1_r = list[i][0] + offset / math.cos((alpha2-alpha1)/2) * math.cos(alpha_r_of)
            y1_r = list[i][1] + offset / math.cos((alpha2-alpha1)/2) * math.sin(alpha_r_of)

            x1_l = list[i][0] + offset / math.cos((alpha2-alpha1)/2) * math.cos(alpha_l_of)
            y1_l = list[i][1] + offset / math.cos((alpha2-alpha1)/2) * math.sin(alpha_l_of)
            
            points_right.append([x1_l ,y1_l])
            points_left.append([x1_r ,y1_r])

            if  1:
                x.append(list[i][0])
                y.append(list[i][1])
            x_r.append(x1_r)
            y_r.append(y1_r)
            x_l.append(x1_l)
            y_l.append(y1_l)
        
    return points_right,points_left

def extract_data_and_convert_to_useful_info(file_path, silent=False):
    extracted_data = []

    i = 0
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        error_gps = 0

        for row in reader:
            i = i + 1
            error_gps = error_gps + float(row['gps_err'])
            lat = float(row['lat'])
            long = float(row['long'])
            x, y = convert_lat_long_to_utm(lat, long)

            imu_orient_x = float(row['imu_orient_x'])
            imu_orient_y = float(row['imu_orient_y'])
            imu_orient_z = float(row['imu_orient_z'])
            imu_orient_w = float(row['imu_orient_w'])
            lin_velocity = float(row['linear_velocity(m/s)'])
            angular_velocity = float(row['imu_ang_vel_z(rad/s)'])
            time = row['time']

            # distance_to_path = float(row['distance_to_path'])
            # if False:#  True and abs(distance_to_path)<100:# abs(angle_to_path)<math.pi/10:
            #     if distance_to_path > 0.4 :
            #         np.array(eval(row['path']))[9][0] 
            #         array= [ np.array(eval(row['path']))[9],np.array(eval(row['path']))[11],np.array(eval(row['path']))[12]]
            #         width = abs(distance_to_path)
            #         points_right,points_left = offset(array,width)
            #         global_path_x = points_left[0][0] + x 
            #         global_path_y = points_left[0][1] + y 

            #     elif  distance_to_path < 0.4:
            #         np.array(eval(row['path']))[9][0] 
            #         array= [ np.array(eval(row['path']))[9],np.array(eval(row['path']))[11],np.array(eval(row['path']))[12]]
            #         width = abs(distance_to_path)
                    
            #         points_right,points_left = offset(array,width)
            #         global_path_x = points_right[0][0] + x 
            #         global_path_y = points_right[0][1] + y 
            # else:
            #     if len(row['path']):
            #         global_path_x = np.array(eval(row['path']))[12][0] + x 
            #         global_path_y = np.array(eval(row['path']))[12][1] + y
            #     else:
            #         global_path_x = x + 5
            #         global_path_y = y + 5

            roll, pitch, yaw = euler_from_quaternion(imu_orient_x, imu_orient_y, imu_orient_z, imu_orient_w)

            extracted_row = {'front_camera': row['front_camera'],
                            'x': x, 'y': y,
                            'yaw': yaw, 'roll': roll, 'pitch': pitch,
                            # 'global_path_x': global_path_x, 'global_path_y': global_path_y,
                            'linear_velocity': lin_velocity,
                            'angular_velocity': angular_velocity,
                            'time': time,
                            }
            
            # optional: fusion data
            fusiondata_keys = ['pose_pos_x', 'pose_pos_y', 'pose_pos_z', 'pose_ori_x', 'pose_ori_y', 'pose_ori_z', 'pose_ori_w']
            if all(key in row.keys() for key in fusiondata_keys):
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
                    'pose_ori_x': float(row['pose_ori_x']),
                    'pose_ori_y': float(row['pose_ori_y']),
                    'pose_ori_z': float(row['pose_ori_z']),
                    'pose_ori_w': float(row['pose_ori_w']),
                    'fusion_yaw': fusion_yaw,
                })

            extracted_data.append(extracted_row)
    
    global total_sample, crr_sample
    crr_sample = i
    total_sample = total_sample + crr_sample

    if i > 0 and not silent:
        print(i)
        print("error gps", (error_gps + 0.0001)/i)

    return extracted_data

def transform_coordinates_to_origin_pose(data):
    x_coords = [row['x'] for row in data]
    y_coords = [row['y'] for row in data]

    # Find the coordinates of the first point
    x_origin, y_origin = x_coords[0], y_coords[0]

    # Subtract the coordinates of the first point from all other points
    transformed_x = [x - x_origin for x in x_coords]
    transformed_y = [y - y_origin for y in y_coords]

    return transformed_x, transformed_y

def transform_coordinates_to_origin_global_path(data):
    x_coords = [row['x'] for row in data]
    y_coords = [row['y'] for row in data]
    
    # Find the coordinates of the first point
    x_origin, y_origin = x_coords[0], y_coords[0]
    
    
    global_path_x_origin = [row['global_path_x'] for row in data]
    global_path_y_origin = [row['global_path_y'] for row in data]
    
    # Subtract the coordinates of the first point from all other points
    transformed_global_path_x = [x - x_origin for x in global_path_x_origin]
    transformed_global_path_y = [y - y_origin for y in global_path_y_origin]

    return transformed_global_path_x, transformed_global_path_y

def time_str2float(time):
    """
    Encode time (string in format yyyy_mm_dd_hh_mm_ss_sss) to seconds

    Args
        time : list
            Time values in string format (yy_mm_dd_hh_mm_ss_sss)

    Returns
        list
            Time values in seconds (float)
    """
    time = time.split('_')
    hour = int(time[3])
    minute = int(time[4])
    second = int(time[5])
    millisecond = int(time[6])
    return (hour * 3600 + minute * 60 + second + millisecond / 1000)


def estimate_path_from_yaw(starting_point, data):
    """
    Estimate the robot's path given the starting point, yaw angles, velocity, and time.
    
    Args
        starting_point : tuple or list 
            Starting (x, y) coordinates.
        data : list of dict 
            List of data points, each with 'yaw', 'linear_velocity', and 'time'.
    
    Returns
        list or tuple
            Estimated path as a list of (x, y) coordinates.
    """
    
    x, y = starting_point[0], starting_point[1]
    path = [list(starting_point)]

    for i in range(1, len(data)):
        yaw = data[i]['yaw']
        velocity = data[i]['linear_velocity']
        time = time_str2float(data[i]['time']) - time_str2float(data[i-1]['time'])

        x += velocity * time * np.cos(yaw)
        y += velocity * time * np.sin(yaw)

        path.append([x, y])
    return path

def estimate_path_from_angular_velocity(starting_point, data, approx='linear'):
    """
    Estimate the robot's path given the starting point, angular velocity, velocity, and time.
    
    Args
        starting_point : tuple or list
            Starting (x, y) coordinates.
        data : list of dict
            List of data points, each with 'yaw','linear_velocity', 'angular_velocity', and 'time'.
        approx : str
            Approximation method ('linear' or 'integral').
    
    Returns
        list of tuple
            Estimated path as a list of (x, y) coordinates.
    """
    assert approx in ['linear', 'integral'], f'{approx} is not supported!'

    x, y = starting_point[0], starting_point[1]
    path = [list(starting_point)]
    angle = data[0]['yaw']

    for i in range(1, len(data)):
        velocity = data[i]['linear_velocity']
        angular_velocity = data[i]['angular_velocity']
        time = time_str2float(data[i]['time']) - time_str2float(data[i-1]['time'])

        if approx == 'integral' or angular_velocity != 0:
            if angular_velocity != 0:
                x += velocity * (np.sin(angle+angular_velocity*time) - np.sin(angle))/angular_velocity
                y -= velocity * (np.cos(angle + angular_velocity*time) - np.cos(angle))/angular_velocity
            else:
                x += velocity * time * np.cos(angle)
                y += velocity * time * np.sin(angle)

        elif approx == 'linear':
            x += velocity * time * np.cos(angle)
            y += velocity * time * np.sin(angle)

        path.append([x, y])
        angle += angular_velocity * time
 
    return path


def plot_transformed_coordinates(data):
    transformed_x, transformed_y = transform_coordinates_to_origin_pose(data)

    yaw_rad = [(row['yaw'] * np.cos(abs(row['pitch']))) for row in data]
    
    yaw_rad_old = [(row['yaw'] + 0./180 * np.pi) for row in data]

    # Customize the point size and color
    point_size = 1
    point_color = 'red'

    # Set the size and color of the vectors
    vector_size = 0.5
    vector_color = 'blue'
    
    step = 1

    if False:
        transformed_global_path_x, transformed_global_path_y = transform_coordinates_to_origin_global_path(data)
        
        # green line
        plt.quiver(
            transformed_x[::step], 
            transformed_y[::step], 
            np.cos(yaw_rad_old[::step]), 
            np.sin(yaw_rad_old[::step]), 
            angles='xy', 
            scale_units='xy', 
            scale=vector_size, 
            color='green', 
            label='Pose_old'
        )

        # red line
        plt.scatter(
            transformed_global_path_x[::step], 
            transformed_global_path_y[::step], 
            label='Global path', 
            s=point_size, 
            c=point_color
        )

        # blue line
        yaw_path = estimate_path_from_yaw(
            starting_point=(transformed_x[0], transformed_y[0]), data=data
        )
        plt.scatter(
            [point[0] for point in yaw_path], 
            [point[1] for point in yaw_path], 
            label='Yaw path',
            s=point_size, 
            color='blue'
        )

    # oranges line
    anl_vel_path = estimate_path_from_angular_velocity(
        starting_point=(transformed_x[0], transformed_y[0]), data=data, approx='integral'
    )
    plt.scatter(
        [point[0] for point in anl_vel_path], 
        [point[1] for point in anl_vel_path], 
        label='Angular velocity path (intergral)',
        s=point_size, 
        color='orange'
    )
    
    # optional plot fusion data
    fusiondata_keys = ['pose_pos_x', 'pose_pos_y', 'pose_pos_z', 'pose_ori_x', 'pose_ori_y', 'pose_ori_z', 'pose_ori_w']
    if all(key in data[0].keys() for key in fusiondata_keys):
        fusion_data = np.array([
            (i['pose_pos_x'], i['pose_pos_y']) for i in data
        ])
        origin_fusion = fusion_data[0, :]
        fusion_data = fusion_data - origin_fusion

        fusion_yaw = [(i['fusion_yaw'] + 0./180 * np.pi) for i in data]

        plt.quiver(
            fusion_data[:, 0], 
            fusion_data[:, 1], 
            np.cos(fusion_yaw[::step]), 
            np.sin(fusion_yaw[::step]), 
            angles='xy', 
            scale_units='xy', 
            scale=vector_size, 
            color='brown', 
            label='Fusion path'
        )

    plt.xlabel('Transformed X Coordinate')
    plt.ylabel('Transformed Y Coordinate')
    plt.title('Data Points as Vectors with Yaw Angles')
    plt.legend()
    plt.grid(True)
    # Set x and y axes with the same scale (aspect ratio)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    

class Displaydata:
    def __init__(
        self,
        image_folder_dir, 
        use_cols=['time', 'angular_velocity(rad/s)', 'linear_velocity(m/s)'], 
        font_color=(0, 0, 255), 
        frame_rate=30, 
        fr_size=0.5, 
        img_type='front_camera',
        csv_path=None
    ):
        """
        To visualize image sequence as video

        Args
            image_folder_dir : str
                Directory to the folder containing images to display
            use_cols : list
                List of columns of interest
            font_color : list or tuple
                RGB code
            frame_rate : int
                FPS
            fr_size : int
                frame size
            imp_type : str
                Type of images to display. Only 'front_camera' and 'routed_map' are supported
            data_csv_dir : str
                Directory to final.csv
        """
        self.image_folder_dir = image_folder_dir
        self.use_cols = use_cols
        self.font_color = font_color
        self.frame_rate = frame_rate
        self.fr_size = fr_size
        self.img_type = img_type
        self.csv_path = csv_path
        

    # @staticmethod
    def display_images_as_video(self):
        """
        Display sequence of images as video
        """

        assert self.img_type in ['front_camera', 'routed_map'], f'{self.img_type} has not been supported yet!'
        assert self.csv_path, 'Please provide the path to the .csv file!'

        time.sleep(0.5)
        images = pd.read_csv(self.csv_path)[self.img_type].tolist()
        images.sort()

        data = self.read_data() if self.csv_path else None
        
        displayed_columns = [col for col in self.use_cols if col!='time']

        print(f"Displaying: {self.image_folder_dir}") 
        print("Press ESC to close the video \n")


        for idx, image in enumerate(images):
            img_path = os.path.join(self.image_folder_dir, image)
            frame = cv2.imread(img_path)   
            frame = cv2.resize(frame, (0, 0), fx=self.fr_size, fy=self.fr_size)

            if self.csv_path: # display the velocity on the image
                text = []
                for i, col in enumerate(displayed_columns):
                    text.append(f"{col}: {data[i+1][idx]:.4f}")

                line_height = 20 
                for i, line in enumerate(text):
                    y_position = 30 + i*line_height 
                    cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.font_color, 2)

            cv2.imshow(self.img_type, frame)
            if cv2.waitKey(int(1000/self.frame_rate)) & 0xFF == 27:
                break

        time.sleep(1)
        cv2.destroyAllWindows()
        matplotlib.use('TkAgg')  # Switch to appropriate GUI backend to prevent Segmentation fault

    def convert_unit(self, dataframe):
        """
            Convert all columns (in m/s) of data frame to km/h        
        """
        for col in [c for c in dataframe.columns if 'm/s' in c]:
            dataframe[col] *= 3.6
            dataframe.rename(columns={col: col.replace('m/s', 'km/h')}, inplace=True)
        
        self.use_cols = list(map(lambda x: x.replace('m/s', 'km/h') if 'm/s' in x else x, 
                                 self.use_cols))
        return dataframe

    def read_data(self):
        """
        Read from the csv file and display the max, min, and mean values of each column
        """
        import pandas as pd
        df = pd.read_csv(self.csv_path, usecols=self.use_cols) 

        df = self.convert_unit(df)
        
        print(f"{' ':<25} {'max':>10} {'min':>10} {'mean':>10}")
        for col in [col for col in self.use_cols if col!='time']:
            print(f'{col:<25} {df[col].max():>10.4f} '
                  f'{df[col].min():>10.4f} {df[col].mean():>10.4f}')

        result = [df[col] for col in self.use_cols]
        return result

    def times_tr2float(self, time):
        """
        Conver time in str (yy_mm_dd_hh_mm_ss_sss) of  to float number
        """
        time = time.split('.')[:-1]
        time = time[0].split('_')
        time = time[-7:]
        year, month, day, hour, minute, second, millisecond = [int(i) for i in time]

        return (year*31536000 + month*2592000 + day*86400 + 
                hour*3600 + minute*60 + second + millisecond/1000)
    
    def frame_lookuptable(self, images, finalcsv_frames):
        """
        
        Args:
            images : list
            finalcsv_frames : list
        """
        lut = {}
        for i in images:
            key = finalcsv_frames[0]
            diff = abs(self.times_tr2float(finalcsv_frames[0]) - self.times_tr2float(i))
            for j in finalcsv_frames:
                temp_diff = abs(self.times_tr2float(j) - self.times_tr2float(i))
                if temp_diff < diff:
                    diff = temp_diff
                    key = j
            lut[i] = j
        
        return lut

class WrongDataHandler:
    @staticmethod
    def append_or_update(data_dir, comment, log_path='wrong_data.log', update=False):
        """
            Append or update the comment to the log file
        """
        line = f"{data_dir}\t{comment}\n"
        
        with open(log_path, 'a') as f:
            if update:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                with open(log_path, 'w') as f:
                    for line_ in lines:
                        if data_dir in line_:
                            f.write(line)
                        else:
                            f.write(line_)
            else:
                f.write(line) 
        f.close()

    @staticmethod
    def write_to_log(data_dir, log_path='wrong_data.log'):
        """
        Write directory to the log file.
        Require user to wrire a comment, along with each directory, to the log file

        Args
            data_dir: directory that will be written to the log file
            log_path: directory to the log file
            
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Path \"{data_dir}\" is not available.")
        
        comment = input("Comment (just enter for no comment): ")
        
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write(f"{data_dir} \t {comment}\n")
            f.close()
        else:
            if data_dir in open(log_path).read():
                print(f"Dirrectory \"{data_dir}\" already exists in {log_path}")
                if input("Do you want to update the comment? y/n \n") == 'y':
                    WrongDataHandler.append_or_update(data_dir, comment, log_path, update=True)
            else:
                WrongDataHandler.append_or_update(data_dir, comment, log_path, update=False)


def check_imu_file(imu_file_path):
    """
    Check if the imu file is available
    """
    if not os.path.exists(imu_file_path):
        raise FileNotFoundError(f"IMU file \"{imu_file_path}\" is not available.")

    df = pd.read_csv(imu_file_path)
    print(df.head(10))
    print(f'imu.csv size: {os.path.getsize(imu_file_path)/1024:2f} kB')

class AutoDetector:
    def __init__(self, detector: discontinuity_detector_pro.InterpolationBasedDetector):
        self.detector = detector

    def check_single_record(self):
        ...

    def auto_check_data(self, data_dir: str, log_path: str = 'autodetected_wrong_data.log') -> tuple:
        """
        Auto-check data to detect discontinuities. The detected error files will be written to the log file.

        Args:
            data_dir : str
                Path to the data directory
                Eg: /media/asimovsimpc/bulldog/aa-data/extracted_data/alaska/umtn-tele/ocp/mix/2024_05_10
            log_path : str
                Path to the log file where the error files will be written

        Returns:
            numb_files : int
                Number of files checked
            numb_detected : int
                Number of error files detected
        """
        with open(log_path, 'a') as f:
            f.close()

        files_to_check = []
        for root, dirs, files in os.walk(data_dir):
            for dir in sorted(dirs):
                if any([x in dir for x in ['2023', '2024']]):
                    files_to_check.append(os.path.join(root, dir, 'final.csv'))

        numb_files = len(files_to_check)
        numb_detected = 0
        for csv_path in tqdm(files_to_check, desc='Checking files', total=numb_files, unit='files'):
            data = extract_data_and_convert_to_useful_info(
                file_path=csv_path, silent=True
            )
            if not data:
                message = "Data hasn't been (or cannot be) extracted\n"

            else:
                data = np.array(transform_coordinates_to_origin_pose(data)).T
                trajectory = data  # Assign the transformed data to trajectory
                discontinuities = self.detector.detect_discontinuities(data)

                if np.sum(discontinuities) < 30:
                    continue
                errors = self.detector.curve_fitting_errors(trajectory)  # Calculate the errors
                above_threshold = np.where(errors > self.detector.threshold)[0]  # Define above_threshold
                # Get RMSE and coefficients for each detected discontinuity
                rmse_values = []
                coefficients = []
                for i in range(0, len(above_threshold)):
                    start = above_threshold[i] * self.detector.step_size
                    end = start + self.detector.window_size
                    window = trajectory[start:end]
                    coeff_x, coeff_y, rmse = self.detector.interpolate(window)

                    x = window[:, 0]
                    y = window[:, 1]

                    # Calculate rmse_x and rmse_y here
                    coeff_x = np.atleast_1d(coeff_x)
                    coeff_y = np.atleast_1d(coeff_y)

                    y_interpolated = np.polyval(coeff_y, x)
                    rmse_y = np.sum((y - y_interpolated) ** 2) / len(y)
                    x_interpolated = np.polyval(coeff_x, y)
                    rmse_x = np.sum((x - x_interpolated) ** 2) / len(x)

                    rmse_values.append(rmse)
                    coefficients.append(coeff_x if rmse_x < rmse_y else coeff_y)

                message = self.detector.read_discontinuities(discontinuities)
                message += f" RMSE values: {rmse_values}, Coefficients: {coefficients}"  # Add to the log message

            csv_path = os.path.dirname(csv_path)
            WrongDataHandler.append_or_update(
                data_dir=csv_path,
                comment=message,
                update=True if csv_path in open(log_path).read() else False,
                log_path=log_path
            )
            numb_detected += 1
        return numb_files, numb_detected
class HumanVerifier:
    def __init__(self, detector: discontinuity_detector_pro.InterpolationBasedDetector):
        self.detector = detector        

    def extract_interval_from_comment(self, comment: str):
        intervals = re.findall(r'\((\d+), (\d+)\)', comment)
        intervals = [(int(start), int(end)) for start, end in intervals]

        # Extract RMSE values and coefficients from the comment
        rmse_values = re.findall(r'RMSE values: \[(.*?)\]', comment)
        coefficients_match = re.findall(r'Coefficients: \[(.*?)\]', comment)

        if rmse_values and coefficients_match:
            rmse_values = [float(x) for x in rmse_values[0].split(', ')]

            # Use regular expressions to extract coefficients
            coefficients_str = coefficients_match[0]
            coefficients = []
            for match in re.finditer(r'\[([^\]]+)\]', coefficients_str):
                coeff_str = match.group(1)
                coeff = np.fromstring(coeff_str, dtype=float, sep=' ')
                coefficients.append(coeff)

        else:
            rmse_values = []
            coefficients = []

        return intervals, rmse_values, coefficients

    def verify_single_record(self, data_dir: str, comment: str = None) -> None:
        """
        Verify a single data directory
        """
        csv_file_path = os.path.join(data_dir, 'final.csv')
        extracted_data_utm = extract_data_and_convert_to_useful_info(csv_file_path)

        intervals, rmse_values, coefficients = self.extract_interval_from_comment(comment)

        # Plot the trajectory error for the entire trajectory
        trajectory = np.array(transform_coordinates_to_origin_pose(extracted_data_utm)).T
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns for subplots

        # Plot 1: Trajectory Error
        self.detector.plot_trajectory_error(
            trajectory,
            point_size=1.5,
            cmap='inferno',
            colorbar=True,
            ax=axes[0]
        )
        axes[0].set_title("Trajectory Error")

        # Plot 2: Trajectory Error with Fitted Curve
        self.detector.plot_trajectory_error(
            trajectory,
            point_size=1.5,
            cmap='inferno',
            colorbar=True,
            ax=axes[1]
        )
        self.detector.plot_fitting_curve(trajectory, index=0, ax=axes[1])
        # Add a message to the title indicating if discontinuities were found
        '''
        if intervals:
            title = "Trajectory Error with Fitted Curve (Discontinuities Detected)"
            axes[1].set_title(title)
        else:
            title = "Trajectory Error with Fitted Curve (No Discontinuities Detected)"
            axes[1].set_title(title)
        '''
        plt.tight_layout()
        plt.show()

#           plt.figure()                                   Original_ha
#            for i in range(4):
#                if not displayed_intervals:
#                    break
#
#               trajectory = np.array(transform_coordinates_to_origin_pose(
#                    extracted_data_utm[displayed_intervals[0][0]: displayed_intervals[0][1]]
#                )).T
#                self.detector.plot_trajectory_error(
#                    trajectory,
#                    point_size=1.5,
#                    cmap='inferno',
#                    colorbar=True,
#                )
#                displayed_intervals.pop(0)
#            plt.show()

#        trajectory = np.array(transform_coordinates_to_origin_pose(
#                    extracted_data_utm
#                )).T
#        self.detector.plot_trajectory_error(
#            trajectory,
#            point_size=1.5,
#            cmap='inferno',
#            colorbar=True,
#        )
#        plt.show()

    def verify_autocheck(self, log_path: str = 'autodetected_wrong_data.log') -> None:
        """
        Verify the auto-checking results by displaying the error files

        Args:
            log_path : str
                Path to the log file where the error files are written
        """
        jump_to = input('Jump to line: ')
        jump_to = 0 if not jump_to else int(jump_to)

        with open(log_path, 'r') as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            if index < jump_to:
                continue

            data_dir, comment = line.split('\t')
            print(f"Data directory: {data_dir}")
            print(f"Comment: {comment}")
            self.verify_single_record(data_dir, comment)
            key = input("Do you want to keep removing the file? y/n \n")
            if key:
                pass

class ManuallyCheck:
    def __init__(self, frame_config):
        self.frame_config = frame_config

    def display_single_record(self, data_dir: str) -> None:
        """
        Visualize a single data directory
        """
        extracted_data_utm = extract_data_and_convert_to_useful_info(data_dir)

        camera_front_dir = os.path.join(path, dir, 'camera_front')
        data_csv_dir = os.path.join(path, dir, 'speed.csv')
        routed_map_dir = os.path.join(path, dir, 'routed_map',)

        Displaydata(
            image_folder_dir=camera_front_dir, 
            use_cols = ['time', 'joystick_angular_velocity(rad/s)', 'joystick_linear_velocity(m/s)'],
            csv_path=csv_file_path,
            frame_rate=self.frame_config['front_camera']['frame_rate'],
            fr_size=self.frame_config['front_camera']['fr_size'],
            font_color=self.frame_config['front_camera']['font_color'],
        ).display_images_as_video()

        if not extracted_data_utm:
            print(path)
            path_ = os.path.join(path, dir)
            shutil.rmtree(path_)

        plot_transformed_coordinates(extracted_data_utm)
        Displaydata(
            image_folder_dir=routed_map_dir, 
            img_type='routed_map',
            use_cols=['time', 'gps_err'],
            csv_path=csv_file_path,
            frame_rate=self.frame_config['routed_map']['frame_rate'],
            fr_size=self.frame_config['routed_map']['fr_size'],
            font_color=self.frame_config['routed_map']['font_color'],
        ).display_images_as_video()

        check_imu_file(os.path.join(path, dir, 'imu.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Auto-check data')
    parser.add_argument('--autocheck', action='store_true', help="Apply autochecking")
    parser.add_argument('--human_verify', action='store_true', help="Verify the auto-checking results")
    parser.add_argument('--manually_check', action='store_true', help="Manually check the data")    
    args = parser.parse_args()

   
    autocheck_parent_dir= "/home/aa/DE/data/extracted_data/bulldog/umtn-tele-joystick/pnk/2.2"
    detector = discontinuity_detector_pro.InterpolationBasedDetector(
        window_size=60, # old: 30
        threshold=0.005, # old: 0.3 (meter)
        step_size=1,
        order=3 # old: 4 w
    )

    if args.autocheck:
        start_time = time.time()
        numb_files, numb_detected = 0, 0
        sub_dirs = os.listdir(autocheck_parent_dir)
        for i, dir in enumerate(sub_dirs):
            path = os.path.join(autocheck_parent_dir, dir)
            print(f'[{i+1}/{len(sub_dirs)}] {path}')

            numb_files_, numb_detected_ = AutoDetector(detector).auto_check_data(data_dir=path)
            numb_files += numb_files_
            numb_detected += numb_detected_

        print(f"{numb_files} files have been checked. " 
              f"{numb_detected} error files have been detected. \n"
              f"Runtime: {time.time() - start_time:.2f} s")
        os._exit(1)

    elif args.human_verify:
        HumanVerifier(detector).verify_autocheck()
        os._exit(1)

    elif args.manually_check:
        with open('pose_plot_config.json') as f:
            config = json.load(f)
            path = config['path']
            frame_config = config['frame_config']
            # breakpoint()

        jump = True
        jump_to = input('Jump to case (just enter for no jump): ').strip()
        if jump_to == '':
            jump = False
            print('No jump!!')
        
        for (root, dirs, file) in os.walk(path):
            for i, dir in enumerate(sorted(dirs)):
                if any([x in dir for x in ['2023', '2024']]):
                    if jump:
                        if dir == jump_to:
                            jump = False
                        else:
                            print(f'Skip {dir}')
                            continue
                            
                    print(f"\n[{i+1}/{len(dirs)}] visualize file {dir}")
                    csv_file_path = os.path.join(path, dir, 'final.csv')
                    ManuallyCheck(frame_config=frame_config).display_single_record(csv_file_path)
                    
                    key = input("do you want remove file y/n \n")
                    if key == "y":
                        print(path)
                        path_ = os.path.join(path, dir)
                        WrongDataHandler.write_to_log(path_)
                        total_sample = total_sample - crr_sample
                    else:
                        pass
        print(f"Total sample are: {total_sample}")
        os._exit(1)
