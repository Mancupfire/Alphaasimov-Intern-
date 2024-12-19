import os
import json
import argparse
import re
import warnings
#from tqdm import tqdm
from typing import List, Union, Callable

def is_valid_datetime_format(string: str) -> bool:
    """
    Check if the string starts with the format yyyy_mm_dd_hh_mm_ss

    Args
        string : str
            Time value in string format (yyyy_mm_dd_hh_mm_ss)

    Returns
        bool
            True if the time is in the correct format, False otherwise
    """
    pattern = r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'
    match = re.match(pattern, string)
    return bool(match)

def datetime_str2float(datetime: str) -> float:
    """
    Convert datetime (string in format yyyy_mm_dd_hh_mm_ss) to seconds

    Args
        time : list
            Time values in string format (yy_mm_dd_hh_mm_ss)

    Returns
        list
            Time values in seconds (float)
    """
    pattern = r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})'
    match = re.search(pattern, datetime)
    if match:
        datetime = match.group(1)
    components = datetime.split('_')
    hour, minute, second = (int(x) for x in components[3:])
    return hour * 3600 + minute * 60 + second

def find_folders(
        sources: Union[str, List[str]], condition: Callable[[str], bool]
    ) -> List[str]:
    """
    Find all folders that meet the condition in the data source directory

    Args:
        sources: str or list of str
            data source directory
        condition: callable
            condition to be met
    
    Returns:
        list of folder directories
    """
    sources = [sources] if isinstance(sources, str) else sources
    
    target_folders = []
    for source in sources:
        for root, dirs, files in os.walk(source):
            for d in dirs:
                if condition(d):
                    target_folders.append(os.path.join(root, d))

    if not target_folders:
        raise warnings.warn(f'No folders meeting the condition were found in {sources}')
    
    return target_folders

def select_data_record(
        data_source: str, scenario: str, date: str, time: str = '00:00-23:59'
    ) -> List[str]:
    """
    Select list of files to be processed

    Args:
        data_source: str
            data source directory
        scenario: str
            scenario name
        date: str
            date
        time: str 
            time range in the format of 'hh:mm-hh:mm'
        
    Returns:
        list of file directories to be processed
    """
    def convert_time(time: str) -> int:
        components = time.split(':')
        # we expect components to contain hour, minute, and second
        # if not, we pad the list with zeros 
        components += ['00'] * (3 - len(components)) 
        hour, minute, second = (int(x) for x in components)
        return hour * 3600 + minute * 60 + second

    if not time:
        start = '00:00'
        end = '23:59'
    else:
        start, end = time.split('-')
    start, end = convert_time(start), convert_time(end)

    # filter data folder based on scenario and date, and folder name format
    conditions = [
        lambda x: x == scenario,
        lambda x: x == date,
        is_valid_datetime_format,
    ] # order-sensitive conditions
    data_folders = data_source
    for i in range(len(conditions)):
        print(f'\rChecking condition {i+1}/{len(conditions)}', end='', flush=True)
        data_folders = find_folders(sources=data_folders, condition=conditions[i])
        if not data_folders:
            return []
    print('\n', end='')
    # filter data folder based on time range
    is_in_time_range = lambda x: start <= datetime_str2float(x.split('/')[-1]) <= end
    data_folders = [x for x in data_folders if is_in_time_range(x)]
    if not data_folders:
        warnings.warn(f'No data folders that meet the time range were found.')
    return data_folders


def main():
    # parser = argparse.ArgumentParser(description='Choose whether to select files for download or upload')
    # parser.add_argument('action', type=str, choices=['download', 'upload'], 
    #                     help='Specify whether to download or upload data')

    # args = parser.parse_args()
    # action = args.action
    
    with open('config.json') as f:
        config = json.load(f)

    # if action == 'upload':
    #     print('Upload data')
    #     source = config['local_data']
    # elif action == 'download':
    #     print('Download data')
    #     source = config['remote_data']
    # else:
    #     raise ValueError('Invalid action')

    source = config['local_storage']

    # go through the source directory and select files to be processed
    data_list = select_data_record(
        data_source=source['directory'],
        scenario=config['criteria']['scenario'],
        date=config['criteria']['date'],
        time=config['criteria']['time']
    )
    # write the list of files to a log file
    with open(f'selected_data_list.log', 'w') as f:
        for file in data_list:
            f.write(file + '\n')
        
    print(f'{len(data_list)} records have been selected for processing')

if __name__ == '__main__':
    main()