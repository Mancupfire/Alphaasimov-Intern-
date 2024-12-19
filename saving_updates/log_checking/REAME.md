# Lidar and sonar data analysis
Check and fix format of .csv files (some are not consistent). You have two option: fix and write to new files or overwrite to current files.\
Analyze lidar and sonar data. Temporal data is cut into segments in which the data variation is no more than 0.5. You can change this threshold in `utils.py`.

## Usage
```bash
pip install -r requirements.txt
```

Adjust the variable `data_path` in `check_log.py` to the directory containing your data and then run the command.

```bash
python check_log.py
```
