import numpy as np
import pandas as pd
import os 
from utils import Sequence_Analyser, Csv_fomart_corrector



data_path = 'data'   # change the directory to the folder containing the data


# check format and fix if necessary
header_name= "time,ap_ready,ap_state,lfl.x,lfl.z,cmd_vel_x,cmd_vel_z,high_cmd,e_stop,remote_operation_mode_cmd,enable_rc_drive_mode,operation_mode_state,robot_stuck,ap_vel_cmd/linear.x,ap_vel_cmd/angular.z,delivery_state_monitor,drive_tele_ready,drive_ap_ready,drive_mode_state,drive_velocity/linear.x,drive_velocity/angular.z,operation_mode_server_cmd,drive_tele_mode_server_cmd,server_cmd_state,confirmation,open_lid_cmd,store,custumer,contermet_all,contermet_rotate,contermet_today_delivery,contermet_delivery,lat,long,error_gps,error_status,limit_forward_speed,limit_backward_speed,limit_spin_right_speed,limit_spin_left_speed,sonar_left,lidar_front_left,sonar_front_ct,lidar_front_ct,lidar_front_right,sonar_right,Lidar_Back_Ct,vx,vy,vz,wz,ax,ay,az,cur_1,cur_2,cur_3,cur_4,vol_1,vol_2,vol_3,vol_4,width,distance_to_path,temp_xavier,temp_pi,data_usage,rssi,ber,ping"

if not os.path.exists('fixed_data'):
    os.makedirs('fixed_data')

to_newfile = True
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        print(f"Checking {file}...")
        Csv_fomart_corrector.fix_format(os.path.join(data_path, file), header_name, to_newfile=to_newfile, newfile=f"fixed_data/{file}")


# Analyze the data
selected_columns = ['sonar_left', 'lidar_front_left', 'sonar_front_ct', 'lidar_front_ct', 'lidar_front_right', 'sonar_right']

if not os.path.exists('analyzed_data'):
    os.makedirs('analyzed_data')

data_path = 'fixed_data' if to_newfile else data_path
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        data_frame = pd.read_csv(os.path.join(data_path, file))
        print(f"Analysing {file}...")

        if not os.path.exists(f'analyzed_data/{file[7:9]}_{file[9:11]}_{file[11:13]}_{file[13:-4]}'):
            os.makedirs(f'analyzed_data/{file[7:9]}_{file[9:11]}_{file[11:13]}_{file[13:-4]}')

        for column in selected_columns:
            sequences = Sequence_Analyser.get_sequences(data_frame, column)
            classified_sequences = Sequence_Analyser.classify_sequences(sequences)

            Sequence_Analyser.classified_sequences_to_csv(classified_sequences, 
                        f"analyzed_data/{file[7:9]}_{file[9:11]}_{file[11:13]}_{file[13:-4]}/"
                        f"{file[7:9]}_{file[9:11]}_{file[11:13]}_{file[13:-4]}_{column}.csv")

print("Done!")
