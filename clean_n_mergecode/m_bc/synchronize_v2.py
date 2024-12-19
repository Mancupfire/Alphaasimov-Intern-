import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import argparse


# MAIN_FOLDER_PATH = '/media/asimovsimpc/bulldog/aa-data/extracted_data/bulldog/umtn/phenikaa/mix/2023_12_07/2023_12_07_15_37_39'

# parser = argparse.ArgumentParser(description='Synchronize data from extracted folder')
# parser.add_argument('--extract_dir', dest='extract_dir', default=MAIN_FOLDER_PATH, type=str, help='Path to extracted folder')
# parser.add_argument('--save_dir', dest='save_dir', default=MAIN_FOLDER_PATH, type=str, help='final csv file saving directory')
# parser.add_argument('--rate', dest='rate', default=25, type=int, help='Rate of saving data (Hz)')
# args = parser.parse_args()

MIN_SPEED = 0.03
MIN_LIN_VEL={'1.1': 1.0, '1.2': 0.2, '1.3': 0.2,
             '1.4': 0.2, '1.5': 0.2, '1.6': 0.1,
             '2.1': 0.2, '2.2': 0.2, '2.3': 0.1, '2.4': MIN_SPEED,
             '3.1': 0.2, '3.2': 0.2, '3.3': MIN_SPEED,
             '4.1': 0, '4.2': 0, '4.3': 0, '4.4': 0, '4.5': 0,
             '5.1': 0.2, '5.2': 0.2, '5.3': MIN_SPEED,
             '6.1': MIN_SPEED,
             '7.1': -1,
             '8.1': -1, '8.2': -1,
             'mix': MIN_SPEED,
             }

MAX_LIN_VEL={'1.1': 5, '1.2': 2, '1.3': 2,
             '1.4': 2, '1.5': 3, '1.6': 2,
             '2.1': 2.3, '2.2': 2.3, '2.3': 2, '2.4': 2,
             '3.1': 2.3, '3.2': 2.3, '3.3': 2,
             '4.1': 2.3, '4.2': 2.3, '4.3': 2.3, '4.4': 2.3, '4.5': 2.3,
             '5.1': 2.5, '5.2': 2.5, '5.3': 2,
             '6.1': 4,
             '7.1': 2,
             '8.1': 0.3, '8.2': 0.3,
             'mix': 6,
             }

LIN_VEL_LIMIT_TOLERANCE=0.2

def read_and_clean_data(file_name, main_folder_path, timestamps):
    file_path = os.path.join(main_folder_path, f'{file_name}.csv')
    df = pd.read_csv(file_path)
    df = df[(df != 0).any(axis=1)]
    df['timestamp'] = df['time'].apply(lambda formatted_time: datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp())
    timestamps[file_name] = [datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp() for formatted_time in df['time'].tolist()]
    return df

def synchronize(main_folder_path, save_dir, rate, data_type='umtn', scenario='mix'):

    min_timestamp = 0

    if data_type == 'umtn':
        imagepath_pointer = {key: {} for key in ['camera_front', 'camera_left', 'camera_right', 'camera_back', 'camera_traffic_light', 'routed_map']}
        timestamps = {key: [] for key in ['camera_front', 'camera_left', 'camera_right', 'camera_back', 'camera_traffic_light', 'gps', 'speed', 'global_path', 'high_cmd', 'imu', 'routed_map']}
    elif 's-umtn' in data_type:
        imagepath_pointer = {key: {} for key in ['camera_front', 'routed_map']}
        timestamps = {key: [] for key in ['camera_front', 'gps', 'speed', 'global_path', 'high_cmd', 'imu', 'routed_map']}
    elif data_type == 'bc':
        imagepath_pointer = {key: {} for key in ['camera_front', 'camera_left', 'camera_right', 'camera_back', 'camera_traffic_light']}
        timestamps = {key: [] for key in ['camera_front', 'camera_left', 'camera_right', 'camera_back', 'camera_traffic_light', 'gps', 'speed', 'high_cmd']}
    else: # s-bc
        imagepath_pointer = {key: {} for key in ['camera_front']}
        timestamps = {key: [] for key in ['camera_front', 'gps', 'speed', 'high_cmd']}

    # Collect unique timestamps from image filenames 
    
    for subfolder in list(imagepath_pointer.keys()):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg'):
                formatted_time = os.path.splitext(filename)[0][-23:]
                # formatted_time = formatted_time.split("_", 2)[2]
                time = datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp()
                timestamps[subfolder] += [time, ]
                imagepath_pointer[subfolder][time] = filename


    #Read file csv to df
    if data_type == 'umtn':
        df_csv = {key: read_and_clean_data(key, main_folder_path, timestamps) for key in ['gps', 'speed', 'global_path', 'high_cmd', 'imu',]}
    elif 's-umtn' in data_type:
        df_csv = {key: read_and_clean_data(key, main_folder_path, timestamps) for key in ['gps', 'speed', 'global_path', 'high_cmd', 'imu',]}
    elif data_type == 'bc':
        df_csv = {key: read_and_clean_data(key, main_folder_path, timestamps) for key in ['gps', 'speed', 'high_cmd']}
    else: # s-bc
        df_csv = {key: read_and_clean_data(key, main_folder_path, timestamps) for key in ['gps', 'speed', 'high_cmd']}
    
    df_rows = {}

    #Find min timestamp
    min_values = {key: min(values) for key, values in timestamps.items()}
    max_value_key = max(min_values, key=lambda k: min_values[k])
    min_timestamp = min_values[max_value_key]
    
    #Find max timestamp
    max_values = {key: max(values) for key, values in timestamps.items()}
    max_value_key = max(max_values, key=lambda k: max_values[k])
    max_timestamp = max_values[max_value_key]

    #Get all time stamps
    step_size = 1/rate
    # all_timestamps = np.array([timestamp for timestamps_each in timestamps.values() for timestamp in timestamps_each])
    all_timestamps = np.array([round(min_timestamp + i * step_size, 2) for i in range(int((max_timestamp - min_timestamp) / step_size) + 1)])

    # Create an empty DataFrame to store the final data
    # final_df = pd.DataFrame(columns=['time', 'front_camera', 'left_camera', 'right_camera', 'rear_camera', 'top_camera', 'lat', 'long', 'gps_error', 'angular_velocity(rad/s)', 'linear_velocity(m/s)', ])
    
    if data_type == 'umtn':
        final_df = pd.DataFrame(columns=['time', 'front_camera', 'left_camera', 'right_camera', 'rear_camera', 'top_camera', 'routed_map', 'angular_velocity(rad/s)', 
                                        'linear_velocity(m/s)', 'dir_high_cmd', 'lat', 'long', 'gps_err',
                                        'mag', 'imu_orient_x',	'imu_orient_y',	'imu_orient_z',	'imu_orient_w',	'imu_ang_vel_x(rad/s)', 'imu_ang_vel_y(rad/s)',
                                        'imu_ang_vel_z(rad/s)',	'imu_lin_acc_x(m/s^2)',	'imu_lin_acc_y(m/s^2)',	'imu_lin_acc_z(m/s^2)',	'sonar_right(m)', 'lidar_front_right(m)',
                                        'sonar_front_center(m)', 'lidar_front_center(m)', 'lidar_front_left(m)', 'sonar_left(m)', 'sonar_rear_center(m)', 'angle_to_path', 'path',
                                        'distance_to_path', 'width_path', 'full_route',
                                        'session_path'])
    elif 's-umtn' in data_type:
        final_df = pd.DataFrame(columns=['time', 'front_camera', 'routed_map', 'angular_velocity(rad/s)', 
                                        'linear_velocity(m/s)', 'dir_high_cmd', 'lat', 'long', 'gps_err',
                                        'mag', 'imu_orient_x',	'imu_orient_y',	'imu_orient_z',	'imu_orient_w',	'imu_ang_vel_x(rad/s)', 'imu_ang_vel_y(rad/s)',
                                        'imu_ang_vel_z(rad/s)',	'imu_lin_acc_x(m/s^2)',	'imu_lin_acc_y(m/s^2)',	'imu_lin_acc_z(m/s^2)',	'sonar_right(m)', 'lidar_front_right(m)',
                                        'sonar_front_center(m)', 'lidar_front_center(m)', 'lidar_front_left(m)', 'sonar_left(m)', 'sonar_rear_center(m)', 'angle_to_path', 'path',
                                        'distance_to_path', 'width_path', 'full_route',
                                        'session_path'])
    elif data_type == 'bc':
        final_df = pd.DataFrame(columns=['time', 'front_camera', 'left_camera', 'right_camera', 'rear_camera', 'top_camera', 'angular_velocity(rad/s)', 
                                    'linear_velocity(m/s)', 'dir_high_cmd', 'lat', 'long', 'gps_err',
                                    # 'mag', 'imu_orient_x',	'imu_orient_y',	'imu_orient_z',	'imu_orient_w',	'imu_ang_vel_x(rad/s)', 'imu_ang_vel_y(rad/s)',
                                    # 'imu_ang_vel_z(rad/s)',	'imu_lin_acc_x(m/s^2)',	'imu_lin_acc_y(m/s^2)',	'imu_lin_acc_z(m/s^2)',
                                    'sonar_right(m)', 'lidar_front_right(m)',
                                    'sonar_front_center(m)', 'lidar_front_center(m)', 'lidar_front_left(m)', 'sonar_left(m)', 'sonar_rear_center(m)',
                                    'session_path'])
    else: # s-bc
        final_df = pd.DataFrame(columns=['time', 'front_camera', 'angular_velocity(rad/s)', 
                                    'linear_velocity(m/s)', 'dir_high_cmd', 'lat', 'long', 'gps_err',
                                    # 'mag', 'imu_orient_x',	'imu_orient_y',	'imu_orient_z',	'imu_orient_w',	'imu_ang_vel_x(rad/s)', 'imu_ang_vel_y(rad/s)',
                                    # 'imu_ang_vel_z(rad/s)',	'imu_lin_acc_x(m/s^2)',	'imu_lin_acc_y(m/s^2)',	'imu_lin_acc_z(m/s^2)',
                                    'sonar_right(m)', 'lidar_front_right(m)',
                                    'sonar_front_center(m)', 'lidar_front_center(m)', 'lidar_front_left(m)', 'sonar_left(m)', 'sonar_rear_center(m)',
                                    'session_path'])

    timestamps_sorted = {key: sorted(timestamps[key]) for key in list(timestamps.keys())}
    pointer = {key: 0 for key in list(timestamps.keys())}
    images_path =  {key: None for key in list(imagepath_pointer.keys())}

    total_sample = 0.0
    correct_sample = 0.0
    # Iterate through each timestamp
    # for curr_timestamp in sorted(all_timestamps[all_timestamps >= min_timestamp]):
    # for curr_timestamp in tqdm(sorted(all_timestamps[all_timestamps >= min_timestamp]), desc="Processing timestamps"):
    total_idx = len(all_timestamps[all_timestamps >= min_timestamp])

    for idx, curr_timestamp in enumerate(tqdm(sorted(all_timestamps[all_timestamps >= min_timestamp]), desc="Processing timestamps")):
        ms = '{:03d}'.format((int)((curr_timestamp % 1) * 1e3))
        
        time = datetime.datetime.fromtimestamp(int(curr_timestamp)).strftime("%Y_%m_%d_%H_%M_%S_") + ms
        for key in list(timestamps.keys()):
            try:
                while curr_timestamp > timestamps_sorted[key][pointer[key] + 1]:
                    pointer[key] += 1
            except:
                pass
        
        # Find the corresponding csv data
        for key in list(df_csv.keys()):
            try:
                if abs(timestamps_sorted[key][pointer[key]] - curr_timestamp) > abs(timestamps_sorted[key][pointer[key] + 1] - curr_timestamp):
                    df_rows[key] =  df_csv[key][df_csv[key]['timestamp'] == timestamps_sorted[key][pointer[key] + 1]]
                else:
                    df_rows[key] =  df_csv[key][df_csv[key]['timestamp'] == timestamps_sorted[key][pointer[key]]]
            except:        
                df_rows[key] =  df_csv[key][df_csv[key]['timestamp'] == timestamps_sorted[key][pointer[key]]]
        
        total_sample += 1

        if df_rows['speed']['linear_velocity(m/s)'].values[0] <= 0 and (idx < 50 or (idx + 50) > total_idx):
            continue

        # If we have wrong speed of the input scenario -> skip this timestamp
        if df_rows['speed']['linear_velocity(m/s)'].values[0] < MIN_LIN_VEL[scenario]:            
            continue
        elif df_rows['speed']['linear_velocity(m/s)'].values[0] > MAX_LIN_VEL[scenario] + LIN_VEL_LIMIT_TOLERANCE:
            continue
            
        correct_sample += 1
        
        # Find the corresponding image paths
        for key in list(images_path.keys()):
            try:
                if abs(timestamps_sorted[key][pointer[key]] - curr_timestamp) > abs(timestamps_sorted[key][pointer[key] + 1] - curr_timestamp):
                    sub_time = timestamps_sorted[key][pointer[key] + 1]
                else:
                    sub_time = timestamps_sorted[key][pointer[key]]
            except:
                sub_time = timestamps_sorted[key][pointer[key]]
            images_path[key] = imagepath_pointer[key][sub_time]

        # Append data to the final DataFrame
        
        if data_type == 'umtn':
            data = {
                'time': time,
                'front_camera': images_path['camera_front'],
                'left_camera': images_path['camera_left'],
                'right_camera': images_path['camera_right'],
                'rear_camera': images_path['camera_back'],
                'top_camera': images_path['camera_traffic_light'],
                'routed_map': images_path['routed_map'],
                'angular_velocity(rad/s)': df_rows['speed']['angular_velocity(rad/s)'].values[0],
                'linear_velocity(m/s)': df_rows['speed']['linear_velocity(m/s)'].values[0],
                'dir_high_cmd': df_rows['high_cmd']['dir_high_cmd'].values[0],
                'lat': df_rows['gps']['lat'].values[0],
                'long': df_rows['gps']['long'].values[0],
                'gps_err': df_rows['gps']['gps_err'].values[0],
                'mag': df_rows['imu']['mag'].values[0],
                'imu_orient_x': df_rows['imu']['imu_orient_x'].values[0],
                'imu_orient_y': df_rows['imu']['imu_orient_y'].values[0],
                'imu_orient_z': df_rows['imu']['imu_orient_z'].values[0],
                'imu_orient_w': df_rows['imu']['imu_orient_w'].values[0],
                'imu_ang_vel_x(rad/s)': df_rows['imu']['imu_ang_vel_x(rad/s)'].values[0],
                'imu_ang_vel_y(rad/s)': df_rows['imu']['imu_ang_vel_y(rad/s)'].values[0],
                'imu_ang_vel_z(rad/s)': df_rows['imu']['imu_ang_vel_z(rad/s)'].values[0],
                'imu_lin_acc_x(m/s^2)': df_rows['imu']['imu_lin_acc_x(m/s^2)'].values[0],
                'imu_lin_acc_y(m/s^2)': df_rows['imu']['imu_lin_acc_y(m/s^2)'].values[0],
                'imu_lin_acc_z(m/s^2)': df_rows['imu']['imu_lin_acc_z(m/s^2)'].values[0],
                'sonar_right(m)': -1,
                'lidar_front_right(m)': -1,
                'sonar_front_center(m)': -1,
                'lidar_front_center(m)': -1,
                'lidar_front_left(m)': -1,
                'sonar_left(m)': -1,
                'sonar_rear_center(m)': -1,
                'angle_to_path': df_rows['global_path']['angle_to_path'].values[0],
                'path': df_rows['global_path']['path'].values[0],
                'distance_to_path': df_rows['global_path']['distance_to_path'].values[0], 
                'width_path': df_rows['global_path']['width_path'].values[0],
                'full_route': [],        
            }
        elif 's-umtn' in data_type:
            data = {
                'time': time,
                'front_camera': images_path['camera_front'],
                'routed_map': images_path['routed_map'],
                'angular_velocity(rad/s)': df_rows['speed']['angular_velocity(rad/s)'].values[0],
                'linear_velocity(m/s)': df_rows['speed']['linear_velocity(m/s)'].values[0],
                'dir_high_cmd': df_rows['high_cmd']['dir_high_cmd'].values[0],
                'lat': df_rows['gps']['lat'].values[0],
                'long': df_rows['gps']['long'].values[0],
                'gps_err': df_rows['gps']['gps_err'].values[0],
                'mag': df_rows['imu']['mag'].values[0],
                'imu_orient_x': df_rows['imu']['imu_orient_x'].values[0],
                'imu_orient_y': df_rows['imu']['imu_orient_y'].values[0],
                'imu_orient_z': df_rows['imu']['imu_orient_z'].values[0],
                'imu_orient_w': df_rows['imu']['imu_orient_w'].values[0],
                'imu_ang_vel_x(rad/s)': df_rows['imu']['imu_ang_vel_x(rad/s)'].values[0],
                'imu_ang_vel_y(rad/s)': df_rows['imu']['imu_ang_vel_y(rad/s)'].values[0],
                'imu_ang_vel_z(rad/s)': df_rows['imu']['imu_ang_vel_z(rad/s)'].values[0],
                'imu_lin_acc_x(m/s^2)': df_rows['imu']['imu_lin_acc_x(m/s^2)'].values[0],
                'imu_lin_acc_y(m/s^2)': df_rows['imu']['imu_lin_acc_y(m/s^2)'].values[0],
                'imu_lin_acc_z(m/s^2)': df_rows['imu']['imu_lin_acc_z(m/s^2)'].values[0],
                'sonar_right(m)': -1,
                'lidar_front_right(m)': -1,
                'sonar_front_center(m)': -1,
                'lidar_front_center(m)': -1,
                'lidar_front_left(m)': -1,
                'sonar_left(m)': -1,
                'sonar_rear_center(m)': -1,
                'angle_to_path': df_rows['global_path']['angle_to_path'].values[0],
                'path': df_rows['global_path']['path'].values[0],
                'distance_to_path': df_rows['global_path']['distance_to_path'].values[0], 
                'width_path': df_rows['global_path']['width_path'].values[0],
                'full_route': [],        
            }
        elif data_type == 'bc':
            data = {
                'time': time,
                'front_camera': images_path['camera_front'],
                'left_camera': images_path['camera_left'],
                'right_camera': images_path['camera_right'],
                'rear_camera': images_path['camera_back'],
                'top_camera': images_path['camera_traffic_light'],
                'angular_velocity(rad/s)': df_rows['speed']['angular_velocity(rad/s)'].values[0],
                'linear_velocity(m/s)': df_rows['speed']['linear_velocity(m/s)'].values[0],
                'dir_high_cmd': df_rows['high_cmd']['dir_high_cmd'].values[0],
                'lat': df_rows['gps']['lat'].values[0],
                'long': df_rows['gps']['long'].values[0],
                'gps_err': df_rows['gps']['gps_err'].values[0],
                'sonar_right(m)': -1,
                'lidar_front_right(m)': -1,
                'sonar_front_center(m)': -1,
                'lidar_front_center(m)': -1,
                'lidar_front_left(m)': -1,
                'sonar_left(m)': -1,
                'sonar_rear_center(m)': -1,       
            }
        else:
            data = {
                'time': time,
                'front_camera': images_path['camera_front'],
                'angular_velocity(rad/s)': df_rows['speed']['angular_velocity(rad/s)'].values[0],
                'linear_velocity(m/s)': df_rows['speed']['linear_velocity(m/s)'].values[0],
                'dir_high_cmd': df_rows['high_cmd']['dir_high_cmd'].values[0],
                'lat': df_rows['gps']['lat'].values[0],
                'long': df_rows['gps']['long'].values[0],
                'gps_err': df_rows['gps']['gps_err'].values[0],
                'sonar_right(m)': -1,
                'lidar_front_right(m)': -1,
                'sonar_front_center(m)': -1,
                'lidar_front_center(m)': -1,
                'lidar_front_left(m)': -1,
                'sonar_left(m)': -1,
                'sonar_rear_center(m)': -1,       
            }

        # data['session_path'] = save_dir.split('/')[-4] + '/' \
        #             + save_dir.split('/')[-3] + '/' \
        #             + save_dir.split('/')[-2] + '/' \
        #             + save_dir.split('/')[-1]

        data['session_path'] = os.sep.join(save_dir.split(os.sep)[-4:])

        row_df = pd.DataFrame([data])
        final_df = pd.concat([final_df, row_df], ignore_index=True)

    # Save the final DataFrame to a CSV file
    final_df.to_csv(os.path.join(save_dir, 'final.csv'), index=False)

    statistic_df = pd.DataFrame(columns=['Total frames', 'Correct frame', 'Correct ratio'])
    data = {'Total frames': total_sample,
            'Correct frame': correct_sample,
            'Correct ratio': f'{(correct_sample/total_sample * 100):.2f}'}
    row_df = pd.DataFrame([data])
    statistic_df = pd.concat([statistic_df, row_df])
    statistic_df.to_csv(os.path.join(save_dir, 'statistic.csv'), index=False)


if __name__ == '__main__':
    synchronize(args.extract_dir, args.save_dir, args.rate)   
    print('SYNCHRONIZING PROCESS COMPLETE')
