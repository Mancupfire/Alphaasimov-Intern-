#!/bin/bash
# DIR="/share-work/rosbag/phenikaa/2023_10_06/*"
# DIR="/media/asimovsimpc/bulldog/aa-data/standardize_behaviour/phenikaa_test/2023_10_23*"
# DIR="/share-work/rosbag/simulation/2023_09_17/*"
# DIR="/share-work/rosbag/2023_09_26/*"
PARENT_DIR="/media/asimovsimpc/bulldog/high_cmd_data/ocp"
EXTRACT_DIR="/media/asimovsimpc/bulldog/aa-data/extracted_data/high_cmd"
# EXTRACT_DIR="/home/asimovsimpc/share-work/extracted_data/high_cmd"
EXTRACT_DATE="*2023_11_14_*.bag"

# rate=2
# fps=22.8
rate=2
fps=25
MIN_SPEED=0.03
CAMERA_NAME="/front"
CAMERA_TOPIC="/zed2i/zed_node/rgb/image_rect_color/compressed"

CHILD_DIR="/1.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=1.5
    MAX_LIN_VEL=3.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/1.2/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.2
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/1.3/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.2
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/1.4/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.2
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/1.5/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.2
    MAX_LIN_VEL=2.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/1.6/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.1
    MAX_LIN_VEL=2.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/2.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=2.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/2.2/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/2.3/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.1
    MAX_LIN_VEL=1.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/2.4/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/3.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/3.2/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/3.3/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.1
    MAX_LIN_VEL=1.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/4.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.9
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/4.2/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.7
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/4.3/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.7
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/4.4/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.7
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/4.5/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.9
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/5.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/5.2/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=0.3
    MAX_LIN_VEL=1.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/5.3/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=1.0
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;

CHILD_DIR="/6.1/*"
DIR="$PARENT_DIR$CHILD_DIR"
echo $DIR
for i in $(find $DIR -name $EXTRACT_DATE);
do
    echo $i
    MIN_LIN_VEL=$MIN_SPEED
    MAX_LIN_VEL=2.5
    /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
    /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
    --bag $i --rate $rate --fps $fps --max_lin_vel $MAX_LIN_VEL --min_lin_vel $MIN_LIN_VEL \
    --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
    sleep 1
    rosbag play -r $rate $i
    sleep 1
    rosnode kill --all
done;