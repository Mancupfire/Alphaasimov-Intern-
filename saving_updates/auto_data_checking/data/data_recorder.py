import pandas as pd
import os
from typing import Dict, List, Any

class DataRecorder:
    """
    Label and record the data.
    """

    @staticmethod
    def label_data() -> Dict[str, int]:
        """
        Ask user and label the data
        """
        key = input('Green and blue lines mismatched? (y/n):\n')
        label = {}
        label['mistmatch'] = 1 if key == 'y' else 0
        
        key = input('Green line discontinuous? (y/n):\n')
        label['discontinuous'] = 1 if key == 'y' else 0
        return label

    @staticmethod
    def write_to_csv(
        trajectory: List[Dict[str, any]],  
        label: Dict[str, int], 
        output_filename: str,
        columns: List[str] = ['time', 'x', 'y', 'global_path_x', 'global_path_y',\
                              'linear_velocity', 'angular_velocity','yaw'],
        origin_path: str = None
    ) -> None:
        """
        Write trajectory data with labels to a CSV file.

        Args
            trajectory : list of dict 
                Trajectory points, each containing 'x', 'y', 'global_path_x', 
                'global_path_y', 'yaw', 'linear_velocity', 'angular_velocity', 'time'.
            label : dict 
                Labels to add, e.g., {'mismatch': 1, 'discontinuous': 0}.
            output_filename : str 
                Path to the output CSV file.
            columns : list of str, optional
                Columns for the CSV. Default is ['x', 'y', 'global_path_x', 
                'global_path_y', 'yaw'].
            origin_path : str, optional
                Path to the original data file.
        """
        trajectory[0].update(label) 
        trajectory[0]['origin_path'] = origin_path
        df = pd.DataFrame(
            trajectory, columns=columns + list(label.keys()) + ['origin_path']
            )

        df.to_csv(output_filename, index=False)

    @staticmethod
    def record(
        output_filename: str, 
        trajectory: List[Dict[str, Any]], 
        folder: str = 'recorded_data',
        origin_path: str = None
    ) -> None:
        """
        Ask user to label the data, then record and save it to data_dir.
        """
        key = input('Record the data? (y/n):\n')
        if key != 'y':
            print('Data not recorded.')
            return

        label = DataRecorder.label_data()
        if not os.path.exists(folder):
            os.makedirs(folder)

        DataRecorder.write_to_csv(
            trajectory=trajectory, 
            label=label, 
            output_filename=os.path.join(folder, output_filename),
            origin_path=origin_path
        )
        

    @staticmethod
    def name_file(csv_dir: str) -> str:
        """
        Generate a file name from the directory of raw data, allowing easy tracking.
        """
        name = csv_dir.split('/')
        name[-1] = '.csv'
        selected = [-7, -6, -5, -4, -2, -1]
        return '_'.join([name[i] for i in selected])

