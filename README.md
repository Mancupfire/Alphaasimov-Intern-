# extract-n-check-data
Codes to extract data from rosbag and check the validation of data
# Installation:
- Follow the installation guideline from the Embedded Team to set up the correct ROS environment to run this code
- Remember to use these versions of message:
  -  https://github.com/AlphaAsimov/collect_data/tree/rosbag
  -  https://github.com/AlphaAsimov/cargobot_msgs/tree/v1



# Run data auto-checking
- Change variable `autocheck_parent_dir` in `pose_plot.py` to your own directory
- Run command
```bash
python pose_plot.py --autocheck
```

# To verifiy errors identified by autodetector
- Run command
``` bash
python pose_plot.py --human_verify
```
**Note**: You can jump dirrectly to a specific line in `autodetected_wrong_data.log` by reponse to the question at the beginning of the program. Just enter to check from the topmost if you don't want to skip any lines. 

# To conduct a (totally) manual verification
- Run
``` bash
python pose_plot.py --manually_check
```