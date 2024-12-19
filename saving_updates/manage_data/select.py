import os
import json
import re
import warnings
from typing import List, Union, Callable


def is_valid_datetime_format(string: str) -> bool:
    """
    Check if the string is in the format yyyy_mm_dd_hh_mm_ss.

    Args:
        string (str): Time value in string format (yyyy_mm_dd_hh_mm_ss).

    Returns:
        bool: True if the string matches the expected format, False otherwise.
    """
    pattern = r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'
    return bool(re.match(pattern, string))


def datetime_str2float(datetime: str) -> float:
    """
    Convert a datetime string (format: yyyy_mm_dd_hh_mm_ss) to seconds since the start of the day.

    Args:
        datetime (str): Datetime string in the format yyyy_mm_dd_hh_mm_ss.

    Returns:
        float: Time in seconds.
    """
    components = datetime.split('_')
    hour, minute, second = map(int, components[3:])
    return hour * 3600 + minute * 60 + second


def find_folders(sources: Union[str, List[str]], condition: Callable[[str], bool]) -> List[str]:
    """
    Find all folders in the data source directory that meet the specified condition.

    Args:
        sources (Union[str, List[str]]): Source directories to search.
        condition (Callable[[str], bool]): Condition to filter folders.

    Returns:
        List[str]: List of folder directories that meet the condition.
    """
    sources = [sources] if isinstance(sources, str) else sources
    target_folders = []

    for source in sources:
        for root, dirs, _ in os.walk(source):
            for folder in dirs:
                if condition(folder):
                    target_folders.append(os.path.join(root, folder))

    if not target_folders:
        warnings.warn(f'No folders meeting the condition were found in {sources}')
    
    return target_folders


def select_data_record(data_source: str, scenario: str, date: str, time: str = '00:00-23:59') -> List[str]:
    """
    Select a list of files to be processed based on scenario, date, and time range.

    Args:
        data_source (str): Directory containing the data.
        scenario (str): Scenario name.
        date (str): Date in yyyy_mm_dd format.
        time (str): Time range in the format 'hh:mm-hh:mm'. Default is '00:00-23:59'.

    Returns:
        List[str]: List of directories matching the criteria.
    """
    def convert_time(time: str) -> int:
        """
        Convert time in hh:mm or hh:mm:ss format to seconds.

        Args:
            time (str): Time string.

        Returns:
            int: Time in seconds.
        """
        components = time.split(':') + ['00'] * (3 - len(time.split(':')))  # Ensure hh:mm:ss format
        hour, minute, second = map(int, components)
        return hour * 3600 + minute * 60 + second

    start, end = map(convert_time, time.split('-'))

    # Define filtering conditions for scenario, date, and folder format
    conditions = [
        lambda x: x == scenario,
        lambda x: x == date,
        is_valid_datetime_format,
    ]

    # Filter folders step by step based on the conditions
    data_folders = [data_source]
    for i, condition in enumerate(conditions):
        print(f'\rChecking condition {i + 1}/{len(conditions)}', end='', flush=True)
        data_folders = find_folders(sources=data_folders, condition=condition)
        if not data_folders:
            return []

    print('\n', end='')

    # Further filter folders based on the time range
    is_in_time_range = lambda x: start <= datetime_str2float(x.split('/')[-1]) <= end
    filtered_folders = [folder for folder in data_folders if is_in_time_range(folder)]

    if not filtered_folders:
        warnings.warn('No data folders meeting the time range were found.')

    return filtered_folders


def main():
    """
    Main function to read configuration, select data, and log the results.
    """
    # Load configuration from JSON file
    with open('config.json') as f:
        config = json.load(f)

    source = config['local_storage']

    # Select data files based on criteria in the configuration
    data_list = select_data_record(
        data_source=source['directory'],
        scenario=config['criteria']['scenario'],
        date=config['criteria']['date'],
        time=config['criteria']['time']
    )

    # Write the selected files to a log
    log_file = 'selected_data_list.log'
    with open(log_file, 'w') as f:
        for file in data_list:
            f.write(file + '\n')

    print(f'{len(data_list)} records have been selected for processing and saved to {log_file}')


if __name__ == '__main__':
    main()
