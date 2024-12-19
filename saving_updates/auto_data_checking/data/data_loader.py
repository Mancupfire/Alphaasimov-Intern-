import pandas as pd
from typing import Union, List

class DataLoader:
    """
    Preprocess and load data.
    """

    @staticmethod
    def time_str2float(time: str) -> float:
        """
        Convert time string to float.

        As the time column in data file is written in string format 'yy_mm_dd_hh_mm_ss_sss'; 
        hence, need to convert it to float for further processing.

        Args
            time : str
        
        Returns
            float
        """
        time = time.split('_')
        year, month, day, hour, minute, second, millisecond = [int(i) for i in time]

        return (year*31536000 + month*2592000 + day*86400 + 
                hour*3600 + minute*60 + second + millisecond/1000)
    
    @staticmethod
    def timeseq_str2float(timeseq: Union[pd.Series, List[str]]) -> Union[pd.Series, List[str]]:
        """
        Convert time sequence string to float and then normalize it.
        
        Args
            timeseq : list of str, or data frame column

        Returns
            list or pd.Series
        """
        if isinstance(timeseq, pd.Series):
            assert all(isinstance(i, str) for i in timeseq), \
                "All elements in the Series must be strings"
            start = DataLoader.time_str2float(timeseq.iloc[0])
            return timeseq.apply(DataLoader.time_str2float) - start
        elif isinstance(timeseq, list):
            assert all(isinstance(i, str) for i in timeseq), \
                "All elements in the list must be strings"
            start = DataLoader.time_str2float(timeseq[0])
            return [DataLoader.time_str2float(time) - start for time in timeseq]
        else:
            raise TypeError("Input should be a list of strings or a pandas Series")

    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform preporcessing on the data, including converting time string to float and 
        normalizing spatial coordinates ('x', 'y', 'global_path_x', 'global_path_y'). 
        
        Args
            data : pd.DataFrame
                Data frame, directly loaded from CSV file, containing the trajectory data.
        
        Returns
            pd.DataFrame
        """
        required_columns = ['x', 'y', 'global_path_x', 'global_path_y', \
                            'yaw', 'linear_velocity', 'angular_velocity', 'time']
        if set(data.columns) != required_columns:
            raise ValueError(
                f"Data columns must be {required_columns}. "
                f"Provided columns are {set(data.columns)}"
            )

        data['time'] = DataLoader.timeseq_str2float(data['time'])

        for (col1, col2) in [('x', 'y'), ('global_path_x', 'global_path_y')]:
            data[col1] = data[col1] - data['x'].iloc[0]
            data[col2] = data[col2] - data['y'].iloc[0]

        return data
    


class DataVisualizer:
    """
    Data visualization.
    """
    ...

