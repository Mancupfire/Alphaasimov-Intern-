import os
import pandas as pd
import numpy as np
import datetime

# Global paramerter
main_folder_path = '/home/pc1/Alpha Asimov/process_data/Data_saved'
min_timestamp = 0


# Collect unique timestamps from image filenames
timestamps = {key: [] for key in ['camera_front', 'camera_left', 'camera_right', 'camera_back', 'cam_traffic_light', 'GPS', 'Speed']}
for subfolder in ['Front', 'Left', 'Right', 'Back', 'Top']:
    subfolder_path = os.path.join(main_folder_path, subfolder)
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.jpg'):
            formatted_time = os.path.splitext(filename)[0]
            formatted_time = formatted_time.split("_", 2)[2]
            time = datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp()
            timestamps[subfolder] += [time, ]

# Read GPS data
gps_path = os.path.join(main_folder_path, 'GPS.csv')
gps_df = pd.read_csv(gps_path)
gps_df = gps_df[(gps_df != 0).any(axis=1)]
gps_df['timestamp'] = gps_df['time'].apply(lambda formatted_time: datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp())
timestamps["GPS"] = [datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp() for formatted_time in gps_df['time'].tolist()]


# Read Speed data
speed_path = os.path.join(main_folder_path, 'Speed.csv')
speed_df = pd.read_csv(speed_path)
speed_df = speed_df[(speed_df != 0).any(axis=1)]
speed_df['timestamp'] = speed_df['time'].apply(lambda formatted_time: datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp())
timestamps["Speed"] = [datetime.datetime.strptime(formatted_time, "%Y_%m_%d_%H_%M_%S_%f").timestamp() for formatted_time in speed_df['time'].tolist()]


all_timestamps = np.array([timestamp for timestamps_each in timestamps.values() for timestamp in timestamps_each])

#Find min timestamp
min_values = {key: min(values) for key, values in timestamps.items()}
max_value_key = max(min_values, key=lambda k: min_values[k])
min_timestamp = min_values[max_value_key]

# Create an empty DataFrame to store the final data
final_df = pd.DataFrame(columns=['time', 'front_camera', 'left_camera', 'right_camera', 'rear_camera', 'top_camera', 'lat', 'long', 'gps_error', 'angular_velocity(rad/s)', 'linear_velocity(m/s)', ])

timestamps_sorted = {key: sorted(timestamps[key]) for key in ['Front', 'Left', 'Right', 'Back', 'Top', 'GPS', 'Speed']}
pointer = {key: 0 for key in ['Front', 'Left', 'Right', 'Back', 'Top', 'GPS', 'Speed']}
images_path =  {key: None for key in ['Front', 'Left', 'Right', 'Back', 'Top']}

# Iterate through each timestamp
for curr_timestamp in sorted(all_timestamps[all_timestamps >= min_timestamp]):
    ms = '{:03d}'.format((int)((curr_timestamp % 1) * 1e3))
    
    time = datetime.datetime.fromtimestamp(int(curr_timestamp)).strftime("%Y_%m_%d_%H_%M_%S_") + ms
    for key in ['Front', 'Left', 'Right', 'Back', 'Top', 'GPS', 'Speed']:
        try:
            while curr_timestamp > timestamps_sorted[key][pointer[key] + 1]:
                pointer[key] += 1
        except:
            pass

    # Find the corresponding GPS data
    gps_row = gps_df[gps_df['timestamp'] == timestamps_sorted['GPS'][pointer['GPS']]]

    # Find the corresponding Speed data
    speed_row = speed_df[speed_df['timestamp'] == timestamps_sorted['Speed'][pointer['Speed']]]
    
    # Find the corresponding image paths
    for key in ['Front', 'Left', 'Right', 'Back', 'Top']:
        images_path[key] = os.path.join(main_folder_path, key, f'{int(timestamps_sorted[key][pointer[key]])}.png')

    # Append data to the final DataFrame
    data = {
        'time': time,
        'front_camera': images_path['Front'],
        'left_camera': images_path['Left'],
        'right_camera': images_path['Right'],
        'rear_camera': images_path['Back'],
        'top_camera': images_path['Top'],
        'lat': gps_row['lat'].values[0],
        'long': gps_row['long'].values[0],
        'gps_error': gps_row['gps_err'].values[0],
        'angular_velocity(rad/s)': speed_row['angular_velocity(rad/s)'].values[0],
        'linear_velocity(m/s)': speed_row['linear_velocity(m/s)'].values[0],
    }

    row_df = pd.DataFrame([data])
    final_df = pd.concat([final_df, row_df], ignore_index=True)


# Save the final DataFrame to a CSV file
final_df.to_csv(os.path.join(main_folder_path, 'final.csv'), index=False)
