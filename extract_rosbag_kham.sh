#!/bin/bash
DIR="/share-work/rosbag/datvt/*"
rate=2

for i in $(find $DIR -name '*.bag');
do
    echo $i   
    /share-work/process_data/safety_rosbag.py & 
    /share-work/process_data/collect_data_example_datvt.py --bag $i --rate $rate &
    sleep 1
    rosbag play -r $rate $i
    # rosbag play $i &
    sleep 1
    rosnode kill --all
done;

