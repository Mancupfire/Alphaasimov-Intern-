import argparse
import glob
import numpy as np
import pandas as pd
import utm


class Columns:
    """
    Column names for the report.
    """
    FILE_PATH = "file_path"
    YAW_PRECISION_RATIO = "yaw_precision_ratio (%)"
    DIFF_YAW_MAX = "diff_yaw_max (deg)"
    DIFF_YAW_MEAN = "diff_yaw_mean (deg)"
    DIFF_YAW_STD = "diff_yaw_std (deg)"


def get_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Yaw Precision Analysis")
    parser.add_argument("-f", "--folder", help="Folder path containing CSV files", type=str, required=True)
    parser.add_argument("-s", "--step", help="Step size for sampling rows", type=int, default=1)
    parser.add_argument("-d", "--diff-yaw-in-deg", help="Yaw difference threshold in degrees", type=float, default=20)
    return parser.parse_args()


def normalize_yaw(yaw: np.ndarray) -> np.ndarray:
    """
    Normalize yaw values to the range [-π, π].
    
    Args:
        yaw (np.ndarray): Array of yaw values.

    Returns:
        np.ndarray: Normalized yaw values.
    """
    return np.arctan2(np.sin(yaw), np.cos(yaw))


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple:
    """
    Convert quaternion values to Euler angles (roll, pitch, yaw).
    
    Args:
        x (float): Quaternion x.
        y (float): Quaternion y.
        z (float): Quaternion z.
        w (float): Quaternion w.

    Returns:
        tuple: (roll, pitch, yaw) in radians.
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return roll, pitch, yaw


def convert_latlon_to_utm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert latitude and longitude to UTM coordinates.
    
    Args:
        df (pd.DataFrame): DataFrame with 'lat' and 'long' columns.

    Returns:
        pd.DataFrame: DataFrame with added 'x' and 'y' columns (UTM coordinates).
    """
    df["x"], df["y"], _, _ = utm.from_latlon(df["lat"].values, df["long"].values)
    return df


def convert_quaternion_to_yaw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert quaternion values to yaw and add as a new column.
    
    Args:
        df (pd.DataFrame): DataFrame with quaternion columns.

    Returns:
        pd.DataFrame: DataFrame with added 'yaw' column.
    """
    df["yaw"] = [
        euler_from_quaternion(row["imu_orient_x"], row["imu_orient_y"], 
                              row["imu_orient_z"], row["imu_orient_w"])[2]
        for _, row in df.iterrows()
    ]
    return df


def get_direction_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate direction angle (dyaw) from UTM coordinates.
    
    Args:
        df (pd.DataFrame): DataFrame with 'x' and 'y' columns.

    Returns:
        pd.DataFrame: DataFrame with added 'dyaw' column.
    """
    df["dx"] = np.diff(df["x"], prepend=df["x"].iloc[0])
    df["dy"] = np.diff(df["y"], prepend=df["y"].iloc[0])
    df["dyaw"] = np.arctan2(df["dy"], df["dx"])
    return df


def check_yaw(df: pd.DataFrame, diff_yaw_in_deg: float) -> tuple:
    """
    Check yaw precision and calculate the valid percentage.
    
    Args:
        df (pd.DataFrame): DataFrame with 'yaw' and 'dyaw' columns.
        diff_yaw_in_deg (float): Threshold for yaw difference in degrees.

    Returns:
        tuple: (Updated DataFrame, precision percentage).
    """
    df["diff_yaw"] = np.abs(normalize_yaw(df["yaw"] - df["dyaw"]))
    valid_mask = df["diff_yaw"] <= np.deg2rad(diff_yaw_in_deg)
    percentage = np.sum(valid_mask) / len(df)
    return df, percentage


def main(args):
    """
    Main function to process CSV files and generate yaw precision report.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    report_data = []

    for csv_path in sorted(glob.glob(f"{args.folder}/*/*.csv")):
        # Preprocess the data
        df = pd.read_csv(csv_path)[::args.step]
        df = convert_latlon_to_utm(df)
        df = convert_quaternion_to_yaw(df)
        df = get_direction_angle(df)

        # Check yaw precision
        df, precision = check_yaw(df, args.diff_yaw_in_deg)
        report_data.append({
            Columns.FILE_PATH: csv_path,
            Columns.YAW_PRECISION_RATIO: precision * 100,
            Columns.DIFF_YAW_MAX: np.rad2deg(df["diff_yaw"].max()),
            Columns.DIFF_YAW_MEAN: np.rad2deg(df["diff_yaw"].mean()),
            Columns.DIFF_YAW_STD: np.rad2deg(df["diff_yaw"].std())
        })

    # Generate and print the report
    report = pd.DataFrame(report_data).sort_values(
        by=[Columns.YAW_PRECISION_RATIO], ascending=False
    ).reset_index(drop=True)
    print(report.to_string())
    # Uncomment to save the report to a CSV file
    # report.to_csv("report.csv", index=False)


if __name__ == "__main__":
    main(get_args())
