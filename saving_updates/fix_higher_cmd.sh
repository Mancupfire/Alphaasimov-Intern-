#!/bin/bash
# DIR="/share-work/rosbag/phenikaa/2023_10_06/*"
DIR="/media/asimovsimpc/bulldog/aa-data/extracted_data/high_cmd/ocp/4.2/*"
# DIR="/share-work/rosbag/simulation/2023_09_17/*"
# DIR="/share-work/rosbag/2023_09_26/*"
value=1

for i in $(find $DIR -name '*2023_11_16_*.csv');
do
    echo $i   
    if [[ "$i" == *"fix"* ]]
    then
        echo "deleting file!"
        rm $i
    else
        echo "updating file..."
        /home/asimovsimpc/share-work/process_data/fix_higher_cmd.py --csv $i --v $value
    fi      
    
done;

for i in $(find $DIR -name '*2023_11_17_*.csv');
do
    echo $i   
    if [[ "$i" == *"fix"* ]]
    then
        echo "deleting file!"
        rm $i
    else
        echo "updating file..."
        /home/asimovsimpc/share-work/process_data/fix_higher_cmd.py --csv $i --v $value
    fi        
    
done;

# DIR="/media/asimovsimpc/bulldog1/aa-data/extracted_data/high_cmd/ocp/4.2/*"
# # DIR="/share-work/rosbag/simulation/2023_09_17/*"
# # DIR="/share-work/rosbag/2023_09_26/*"
# value=1

# for i in $(find $DIR -name '*.csv');
# do
#     echo $i   
#     if [[ "$i" == *"fix"* ]]
#     then
#         echo "deleting file!"
#         rm $i
#     else
#         echo "updating file..."
#         /home/asimovsimpc/share-work/process_data/fix_higher_cmd.py --csv $i --v $value
#     fi
        
    
# done;
