#!/usr/bin/env python3.8
import csv
import os, sys
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Param')
parser.add_argument('--csv', dest='csv_path', \
                default='/home/asimovsimpc/share-work/extracted_data/2023_07_28/2023_07_28_14_14_09/2023_07_28_14_14_09.csv', \
                type=str, help='Path to csv file')
parser.add_argument('--s', dest='start_value', \
                default='2023_07_28_04_20_47_112.jpg', \
                type=str, \
                help='value of search column that locate at the start row need to be changed')
parser.add_argument('--e', dest='end_value', \
                default="2023_07_28_04_21_03_012.jpg", \
                type=str, \
                help='value of search column that locate at the end row need to be changed')
parser.add_argument('--search_col', dest='search_col', \
                        default='front_center', type=str, help='Column name contains key value to get row index')
parser.add_argument('--target_col', dest='target_col', \
                        default="dir_high_cmd", type=str, help='Column name need to be changed')
parser.add_argument('--v', dest='value', \
                    default=1, type=int, help='New value to be updated -- 1: turn left; -1: turn right')
parser.add_argument('--t', dest='type', \
                    default=0, type=int, \
                    help='Fix type: 0-change, 1-delete')                    

if __name__ == '__main__':
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    
    # start_idx = df.index[df[args.search_col] == args.start_value].tolist()    
    # if len(start_idx) > 0:
    #     start_idx = start_idx[0]
    # else:
    #     print("Cannot find idx of the value {} of column {}, exiting...".format(args.start_value, args.search_col))
    #     sys.exit(0)
    
    # end_idx = df.index[df[args.search_col] == args.end_value].tolist()    
    # if len(end_idx) > 0:
    #     end_idx = end_idx[0]
    # else:
    #     print("Cannot find idx of the value {} of column {}, exiting...".format(args.end_idx, args.search_col))
    #     sys.exit(0)

    # print(start_idx+2, end_idx+2)

    df.loc[:, args.target_col] = args.value
    
    # save_path = args.csv_path[:-4] + '_fixed.csv'
    save_path = args.csv_path
    df.to_csv(save_path, index=False)

    