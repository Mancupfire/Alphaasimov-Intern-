import os
import json
import warnings
from typing import List

def find_bag_files(directory: str) -> List[str]:
    """
    Find all .bag files in the given directory and its subdirectories.

    Args:
        directory: str
            The root directory to search for .bag files.

    Returns:
        List of paths to .bag files found.
    """
    bag_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.bag'):
                bag_files.append(os.path.join(root, file))
    
    if not bag_files:
        warnings.warn(f'No .bag files were found in {directory}')
    
    return bag_files

def main():
    # Load configuration
    with open('config.json') as f:
        config = json.load(f)
    
    # Directory from the configuration
    source_directory = config['local_storage']['directory']
    
    # Find all .bag files
    bag_files = find_bag_files(source_directory)
    
    # Write the list of files to a log file
    log_file_path = 'selected_data_list.log'
    with open(log_file_path, 'w') as log_file:
        for bag_file in bag_files:
            log_file.write(bag_file + '\n')
    
    print(f'{len(bag_files)} .bag files have been found and logged to {log_file_path}')

if __name__ == '__main__':
    main()
