#!/usr/bin/env python3.8
import os
import argparse
import rosbag
from tqdm import tqdm

def read_scenario(bag, dataType):
    curr_scenario = ""
    extracted_data = bag.read_messages("/scenario_id")
    id = 0
    start_time = 0
    scenario_dict = {}

    for seq_id, bag_data in tqdm(enumerate(extracted_data), desc="Loading rosbag file"):
        msg = bag_data.message.msg.data
        secs = bag_data.message.header.stamp.secs
        nsecs = bag_data.message.header.stamp.nsecs
        time = secs + nsecs * 1e-9
        if msg != curr_scenario:
            end_time = time
            scenario_dict[id] = {"scenario": curr_scenario, "start_time": start_time, "end_time": end_time}
            start_time = time
            curr_scenario = msg
            id += 1
    scenario_dict[id] = {"scenario": curr_scenario, "start_time": start_time, "end_time": time}
    return scenario_dict

def statistic(scenario_dict):
    for key, value in scenario_dict.items():
        scenario = value['scenario']
        count = value['count']
        if scenario != "":
            print(f"{scenario}: {count} times")

def process_bag_directory(directory, data_type):
    total_scenario_dict = {}
    bag_files = [f for f in os.listdir(directory) if f.endswith(".bag")]

    for bag_file in bag_files:
        bag_path = os.path.join(directory, bag_file)
        print(f"Processing bag file: {bag_path}")
        bag = rosbag.Bag(bag_path)
        scenario_dict = read_scenario(bag, data_type)

        # Accumulate statistics
        for key, value in scenario_dict.items():
            scenario = value['scenario']
            if scenario != "":
                if scenario in total_scenario_dict:
                    total_scenario_dict[scenario]['count'] += 1
                else:
                    total_scenario_dict[scenario] = {'scenario': scenario, 'count': 1}

        bag.close()

    print("\nTotal Statistics:")
    statistic(total_scenario_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract data from rosbag files in a directory')
    parser.add_argument('--dir', dest='bag_directory', default='/home/asimovsimpc/share-work/process_data/m_bc/rosbag_for_test/phenikaa/2023_12_15_a01/', type=str, help='Directory containing bag files')
    parser.add_argument('--rate', dest='rate', default=25, type=int, help='Rate of saving data (Hz)')
    parser.add_argument('--data', dest='data_type', default='umtn', type=str, help='Data type: umtn or bc')
    args = parser.parse_args()

    process_bag_directory(args.bag_directory, args.data_type)
