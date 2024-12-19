#!/bin/bash
# DIR="/share-work/rosbag/phenikaa/2023_10_06/*"
DIR="/media/asimovsimpc/bulldog/aa-data/test_bno/2023_11_17/*"

# DIR="/share-work/rosbag/simulation/2023_09_17/*"
# DIR="/share-work/rosbag/2023_09_26/*"
rate=2
fps=30

for i in $(find $DIR -name '*.bag');
do
    echo $i   
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
    /home/asimovsimpc/share-work/process_data/extract_data_by_rosbag_file_test_imu_camera.py --bag $i --rate $rate --fps $fps &
    sleep 1
    rosbag play -r $rate $i    
    sleep 1
    rosnode kill --all
done;

# DIR="/share-work/rosbag/camera_138/*"

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_rosbag_file_test_imu_camera.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/phenikaa/2.2/2023_10_08_rb02/*"

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario_v3.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/phenikaa/4.1/2023_10_08_rb02/*"

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario_v3.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/phenikaa/4.2/2023_10_08_rb02/*"

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario_v3.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/phenikaa/4.3/2023_10_08_rb02/*"

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario_v3.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/4.2/2023_10_01_rb03/*"
# rate=2
# fps=30

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

# DIR="/share-work/rosbag_scenario/1.5/2023_09_24_rb03_1_5/*"
# rate=2
# fps=30

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file_v2.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;
