#!/bin/bash
# DIR="/share-work/rosbag/phenikaa/2023_10_06/*"
# DIR="/media/asimovsimpc/bulldog/aa-data/standardize_behaviour/phenikaa_test/2023_10_23*"
# DIR="/share-work/rosbag/simulation/2023_09_17/*"
# DIR="/share-work/rosbag/2023_09_26/*"
PARENT_DIR="/media/asimovsimpc/bulldog/high_cmd_data/ocp"
EXTRACT_DIR="/media/asimovsimpc/bulldog/aa-data/extracted_data/high_cmd"
# EXTRACT_DIR="/home/asimovsimpc/share-work/extracted_data/high_cmd_test"
# EXTRACT_DATE="*2023_11_14_*.bag"
EXTRACT_DATE=("*2023_11_17_*.bag")

SCENARIO=(\
"1.1" "1.2" "1.3" "1.4" "1.5" "1.6" \
"2.1" "2.2" "2.3" "2.4" \
"3.1" "3.2" "3.3" \
"4.1" "4.2" "4.3" "4.4" "4.5" \
"5.1" "5.2" "5.3" \
"6.1" \
)

# EXTRACT_SCENARIO=(\
# "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" \
# "2.1" "2.2" "2.3" "2.4" \
# "3.1" "3.2" "3.3" \
# "4.1" "4.2" "4.3" "4.4" "4.5" \
# "5.1" "5.2" "5.3" \
# "6.1" \
# )

EXTRACT_SCENARIO=(\
"4.1" "4.2" \
)

MIN_SPEED=0.03

MIN_LIN_VEL=(\
1.5 0.2 0.2 0.2 0.2 0.1 \
0.3 0.3 0.1 $MIN_SPEED \
0.3 0.3 $MIN_SPEED  \
$MIN_SPEED $MIN_SPEED $MIN_SPEED $MIN_SPEED $MIN_SPEED \
0.3 0.3 $MIN_SPEED \
$MIN_SPEED \
)

MAX_LIN_VEL=(\
3 3 1.5 1.5 2 1.5 \
2 1.5 1 1 \
1.5 1.5 1 \
1.9 1.7 1.7 1.7 1.9 \
1.5 1.5 1 \
2.5 \
)


# rate=2
# fps=22.8
rate=2
fps=25

CAMERA_NAME="/front"
CAMERA_TOPIC="/zed2i/zed_node/rgb/image_rect_color/compressed"

for date in ${EXTRACT_DATE[@]}; do
    for i in ${!SCENARIO[@]}; do
        # DIR="$PARENT_DIR/${SCENARIO[$i]}/*"  
        # echo $DIR
        
        need_extract=false
        for j in ${!EXTRACT_SCENARIO[@]}; do
            if [[ ${EXTRACT_SCENARIO[$j]} == ${SCENARIO[$i]} ]]; then
                need_extract=true
                break
            fi
        done

        if ! $need_extract; then
            continue
        fi

        DIR="$PARENT_DIR/${SCENARIO[$i]}/*"  
        echo "Extracting data from $DIR ..."

        for f in $(find $DIR -name $date); do           
            echo "$f -- id = $i -- max-speed = ${MAX_LIN_VEL[$i]} -- min-speed = ${MIN_LIN_VEL[$i]}"

            /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
            /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd.py \
            --bag $f --rate $rate --fps $fps --max_lin_vel ${MAX_LIN_VEL[$i]} --min_lin_vel ${MIN_LIN_VEL[$i]} \
            --dir $EXTRACT_DIR --cam_name $CAMERA_NAME --cam_topic $CAMERA_TOPIC &
            sleep 1
            rosbag play -r $rate $f
            sleep 1
            rosnode kill --all
        done
    done
done