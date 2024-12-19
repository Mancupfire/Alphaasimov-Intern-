import plotly
import plotly.express as px
import plotly.io as pio
import pandas as pd
import os


def plot_dataframe(df, save_path, height=900, width=900, zoom = 15.5, marker_size=4, center=None, silent=True, title='Collected Data'):
    """
        Plot a map from a dataframe and save.
    """
    fig = px.scatter_mapbox(df,
                            lon=df['long'],
                            lat=df['lat'],
                            zoom = zoom,
                            width = width,
                            height= height,
                            title = title
                            )

    
    fig.update_traces(marker=dict(size=marker_size, color='red'))

    map_center = {
        "lon": center[0] if center is not None else df['long'].mean(),
        "lat": center[1] if center is not None else df['lat'].mean()
    }
    fig.update_layout(mapbox_style="carto-positron",
                      margin={"r":0, "t":50, "l":0, "b":10},
                      mapbox=dict(center=map_center)
                      )
    if not silent:
        fig.show()
        
    pio.write_image(fig, f"{save_path}/{title}.png")



def read_final_csv(data_path, usecols=['long', 'lat']):
    """
        Read all final.csv files in the data_path
        Return:
            Data frame
    """
    final_csv_files = []

    for root, dirs, files in os.walk(data_path, topdown=True):
        for file in files:
            if file.endswith("final.csv"):
                final_csv_files.append(os.path.join(root, file))
            # print(os.path.join(root, file))

    # read all csv file and create data frame from 'long' and 'lat' column
    df = pd.DataFrame()
    for file in final_csv_files:
        print(f"Reading file: {file}")
        sub_df = pd.read_csv(file, usecols=usecols)
        sub_df = sub_df.iloc[:-1:10]
        df = df._append(sub_df, ignore_index=True)

    return df


def plot_scenario(data_path, save_path, scenario, height=900, width=900, zoom = 15.5, marker_size=4, center=None, silent=True):
    """
        Plot a map for a specific scenario
    """
    scenario_path = f"{data_path}/{scenario}"
    df = read_final_csv(scenario_path)
    if df.empty:
        print(f"Data frame is empty for scenario: {scenario}. Check the data path.")
        return
    
    plot_dataframe(df, 
                   save_path, 
                   title=f'Scenario {scenario}', 
                   height=height, 
                   width=width, 
                   zoom = zoom, 
                   marker_size=marker_size,
                   center=center, 
                   silent=silent)


def calculate_center(data_path, usecols=['long', 'lat']):
    """
        Read all final.csv file in data_path. Calculate the center of all the data points.
    """
    print("Calculating center of all data points...")
    final_csv_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("final.csv"):
                final_csv_files.append(os.path.join(root, file))

    long_min = None
    long_max = None
    lat_min = None
    lat_max = None

    for file in final_csv_files:
        print(f"Reading file: {file}")
        sub_df = pd.read_csv(file, usecols=usecols) # take colums 'long' and 'lat' only
        if long_min is None or long_min > sub_df['long'].min():
            long_min = sub_df['long'].min()
        if long_max is None or long_max < sub_df['long'].max(): 
            long_max = sub_df['long'].max()
        if lat_min is None or lat_min > sub_df['lat'].min():
            lat_min = sub_df['lat'].min()
        if lat_max is None or lat_max < sub_df['lat'].max():
            lat_max = sub_df['lat'].max()
    
    return [(long_min + long_max) / 2, (lat_min + lat_max) / 2]