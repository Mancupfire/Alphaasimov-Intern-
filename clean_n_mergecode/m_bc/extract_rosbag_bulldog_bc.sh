#!/bin/bash
# DIR="/share-work/rosbag/phenikaa/2023_10_06/*"
# DIR="/media/asimovsimpc/bulldog/aa-data/standardize_behaviour/phenikaa_test/2023_10_23*"
# DIR="/share-work/rosbag/simulation/2023_09_17/*"
# DIR="/share-work/rosbag/2023_09_26/*"
PARENT_DIR="/media/asimovsimpc/bulldog/aa-data/bulldog/bc/phenikaa"
# PARENT_DIR="/media/asimovsimpc/bulldog/aa-data/rosbag_scenario/eco"
EXTRACT_DIR="/media/asimovsimpc/bulldog/aa-data/extracted_data/bulldog/bc"
# EXTRACT_DIR="/home/asimovsimpc/share-work/extracted_data/high_cmd_test"
# EXTRACT_DATE=("*.bag")
EXTRACT_DATE=("2023_12_05*.bag" "2023_11_21*.bag" "2023_11_24*.bag")
# EXTRACT_DATE=("2023_12_08*.bag")

# DATA_TYPE="umtn"
DATA_TYPE="bc"

SCENARIO=(\
"1.1" "1.2" "1.3" "1.4" "1.5" "1.6" \
"2.1" "2.2" "2.3" "2.4" \
"3.1" "3.2" "3.3" \
"4.1" "4.2" "4.3" "4.4" "4.5" \
"5.1" "5.2" "5.3" \
"6.1" \
"7.1" "7.2" \
"mix"
)

# EXTRACT_SCENARIO=(\
# "1.2" "1.3" "1.4" "1.5" "1.6" \
# "2.1" "2.2" "2.3" "2.4" \
# "3.1" "3.2" "3.3" \
# "4.1" "4.2" "4.3" "4.4" "4.5" \
# "5.1" "5.2" "5.3" \
# "6.1" \
# )

EXTRACT_SCENARIO=(\
# "1.1" "1.2" "1.3" "2.2" "4.1"\
"mix"
)

MIN_SPEED=0.03

MIN_LIN_VEL=(\
0.5 0.2 0.2 0.2 0.2 0.1 \
0.2 0.2 0.1 $MIN_SPEED \
0.2 0.2 $MIN_SPEED  \
$MIN_SPEED $MIN_SPEED $MIN_SPEED $MIN_SPEED $MIN_SPEED \
0.2 0.2 $MIN_SPEED \
$MIN_SPEED \
-1 -1 \
$MIN_SPEED \
)

MAX_LIN_VEL=(\
3 3 1.5 1.5 2 1.5 \
2 1.5 1 1 \
1.5 1.5 1 \
1.9 1.7 1.7 1.7 1.9 \
1.5 1.5 1 \
2.5 \
0.3 0.3 \
6 \
)

fps=20

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

            # /home/asimovsimpc/share-work/process_data/safety_rosbag.py & 
            # /home/asimovsimpc/share-work/process_data/extract_data_by_scenario_v3_high_cmd_for_old_data.py \
            /home/asimovsimpc/share-work/process_data/m_bc/bag_to_all_v2.py \
            --bag $f --dir $EXTRACT_DIR --rate $fps --data $DATA_TYPE
            
            echo "Finish extracting $f"
        done;
    done
done