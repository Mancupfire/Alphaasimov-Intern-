import json
from utils import plot_scenario, calculate_center, read_final_csv, plot_dataframe
import time
import os
import pandas as pd

def process_data_path(data_path, save_path, height=900, width=900, zoom = 15.5, marker_size=4, center=None, silent=False, full_scenarios=False):
    if not os.path.exists(data_path):
        print(f"Path \"{data_path}\" is not available. Skipping...")
        return

    if not os.path.exists(save_path):  # create dir to save results
        os.makedirs(save_path)

    print(f'Analyzing path \"{data_path}\" ...')
    center = calculate_center(data_path) if not center else center
    print("center: ", center)
    if not full_scenarios: # plot indivisual scenarios in the data path
        for scenario in os.listdir(data_path):
            print(f"\nPlotting map for scenario: {scenario}")
            plot_scenario(data_path, 
                        save_path=save_path,
                        scenario=scenario, 
                        height=height, 
                        width=width, 
                        zoom = zoom, 
                        marker_size=marker_size,
                        center=center,
                        silent=silent
                        )
            time.sleep(0.25)

    else: # plot all scenarios in the same map
        df = pd.DataFrame()
        print(f"\nPlotting all scenarios in a single map...")
        for scenario in os.listdir(data_path):
            scenario_path = f"{data_path}/{scenario}"
            if df.empty:
                df = read_final_csv(scenario_path)
            else:
                df = df._append(read_final_csv(scenario_path))
        plot_dataframe(df, 
                    save_path, 
                    title=f'Full Scenarios Map', 
                    height=height, 
                    width=width, 
                    zoom = zoom, 
                    marker_size=marker_size,
                    center=center, 
                    silent=silent)


def main():


    config_file = "config.json"
    with open(config_file) as f:
        config = json.load(f)
        print(f'Loading config file: {config_file} ...')

    data_paths = config['data_paths']
    silent = config['silent']


    result_dir = "results"

    for data_path in data_paths:
        location = config['data_paths'][data_path]['location']
        center = config['frame_config'][location]['center']
        zoom = config['frame_config'][location]['zoom']
        marker_size = config['frame_config'][location]['marker_size']
        frame_height = config['frame_config'][location]['height']
        frame_width = config['frame_config'][location]['width']
        full_scenarios = config['full_scenarios']

        process_data_path(data_path=data_path, 
                          save_path=os.path.join(result_dir, data_path.replace('/', '_')),
                          height=frame_height,
                          width=frame_width,
                          zoom=zoom,
                          marker_size=marker_size,
                          center=center,
                          silent=silent,
                          full_scenarios=full_scenarios)

    print("Done!!!")


if __name__ == "__main__":
    main()