#!/bin/bash
# DIR="/share-work/rosbag/2023_08_15_phenikaa/*"
# DIR="/share-work/rosbag_scenario/*"
DIR="/share-work/rosbag_simulation/2023_09_18/*"
rate=2
fps=30

for i in $(find $DIR -name '*.bag');
do
    echo $i   
    /share-work/process_data/safety_rosbag.py & 
    # /share-work/process_data/extract_data_by_rosbag_file.py --bag $i --rate $rate --fps $fps &
    /share-work/process_data/extract_data_simulation.py --bag $i --rate $rate --fps $fps &
    sleep 1
    rosbag play -r $rate $i
    # rosbag play $i &
    sleep 1
    rosnode kill --all
done;


# DIR="/share-work/rosbag_scenario/1.4/2023_09_10/*"
# rate=2
# fps=30

# for i in $(find $DIR -name '*.bag');
# do
#     echo $i   
#     /share-work/process_data/safety_rosbag.py & 
#     # /share-work/process_data/extract_data_by_rosbag_file.py --bag $i --rate $rate --fps $fps &
#     /share-work/process_data/extract_data_by_scenario.py --bag $i --rate $rate --fps $fps &
#     sleep 1
#     rosbag play -r $rate $i
#     # rosbag play $i &
#     sleep 1
#     rosnode kill --all
# done;

