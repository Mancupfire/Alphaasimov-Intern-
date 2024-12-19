import os
import pandas as pd
from datetime import datetime

# Function to get the number of rows in a CSV file
def get_csv_rows(csv_path):
    df = pd.read_csv(csv_path)
    return len(df)

# Function to process a scenario folder
def process_scenario(scenario_path, report_data):
    total_rows_scenario = 0
    for date_folder in os.listdir(scenario_path):
        date_path = os.path.join(scenario_path, date_folder)
        if os.path.isdir(date_path):
            total_rows_date = 0
            for time_folder in os.listdir(date_path):
                time_path = os.path.join(date_path, time_folder)
                if os.path.isdir(time_path):
                    csv_path = os.path.join(time_path, 'final.csv')
                    if os.path.exists(csv_path):
                        rows = get_csv_rows(csv_path)
                        total_rows_date += rows
                        total_rows_scenario += rows
            report_data.append({
                "Scenario": scenario_path.split('/')[-1],
                "Date": date_folder,
                "Total Rows": total_rows_date
            })

    report_data.append({
        "Scenario": scenario_path.split('/')[-1],
        "Total Rows for Scenario": total_rows_scenario
    })
    return total_rows_scenario

# Main function
def main(main_folder_path):
    report_data = []
    total_data = 0
    for scenario_folder in os.listdir(main_folder_path):
        scenario_path = os.path.join(main_folder_path, scenario_folder)
        if os.path.isdir(scenario_path):
            total_data += process_scenario(scenario_path, report_data)

    report_data.append({
        "Scenario": 'Total data',
        "Total Rows for Scenario": total_data
    })

    report_df = pd.DataFrame(report_data)
    # report_df.to_csv("report_bc.csv", index=False)
    report_df.to_csv("report_umtn.csv", index=False)

if __name__ == "__main__":
    # main_folder_path = "/media/admin02/bulldog2/extracted_data/bulldog/bc/ocp/"  # Update with the actual path to your main folder
    main_folder_path = "/media/admin02/bulldog2/extracted_data/bulldog/umtn/ocp/"  # Update with the actual path to your main folder
    # main_folder_path = "/work/ctv.tindt/bulldog_bc/ocp/"
    main(main_folder_path)
