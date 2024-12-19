import argparse
import glob

import numpy as np
import pandas as pd
import utm


class Columns:
    FILE_PATH = "file_path"
    YAW_PRECISION_RATIO = "yaw_precision_ratio (%)"
    DIFF_YAW_MAX = "diff_yaw_max (deg)"
    DIFF_YAW_MEAN = "diff_yaw_mean (deg)"
    DIFF_YAW_STD = "diff_yaw_std (deg)"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="Folder path", type=str, required=True)
    parser.add_argument("-s", "--step", help="Step to skip samples", type=int, default=1)
    parser.add_argument("-d", "--diff-yaw-in-deg", help="Difference of yaw in degree", type=float, default=20)
    return parser.parse_args()


def normalize_yaw(yaw):
    normalized_yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return normalized_yaw


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float):
    """
    Convert to euler from quaternion
    Args:
        qx (float):
        qy (float):
        qz (float):
        qw (float):

    Returns:
        (roll, pitch, yaw)
    """
    q = np.array([qx, qy, qz, qw])
    # Normalize the quaternion
    q = q / np.linalg.norm(q)

    # Extract quaternion components
    x, y, z, w = q

    # Calculate Euler angles
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return yaw

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def convert_latlon_to_utm(df):
    # convert lat/lon to utm
    df["x"], df["y"], _, _ = utm.from_latlon(df["lat"].values, df["long"].values)
    return df


def convert_quaternion_to_yaw(df):
    # convert to yaw
    df["yaw"] = 0
    for i, row in df.iterrows():
        # df.at[i, "yaw"] = yaw_from_quaternion(row["imu_orient_x"], row["imu_orient_y"],
        #                                       row["imu_orient_z"], row["imu_orient_w"])
        
        _, _, df.at[i, "yaw"] = euler_from_quaternion(row["imu_orient_x"], row["imu_orient_y"],
                                              row["imu_orient_z"], row["imu_orient_w"])
        
    return df


def get_direction_angle(df):
    df["dx"] = np.concatenate([np.diff(df["x"].values), np.array([0])])
    df["dy"] = np.concatenate([np.diff(df["y"].values), np.array([0])])
    df["dyaw"] = np.arctan2(df["dy"].values, df["dx"].values)
    return df


def check_yaw(df, diff_yaw_in_deg):
    """
    Return the true valid percentage
    :param diff_yaw_in_deg:
    :param df:
    :return:
    """
    df["diff_yaw"] = np.abs(normalize_yaw(df["yaw"] - df["dyaw"]))
    mask = df["diff_yaw"] <= np.deg2rad(diff_yaw_in_deg)
    percentage = np.sum(mask) / len(df)
    return df, percentage


def main(args):
    container = []
    for csv_path in sorted(glob.glob(f"{args.folder}/*/*.csv")):
        # preprocess
        df = pd.read_csv(csv_path)[::args.step]
        df = convert_latlon_to_utm(df)
        df = convert_quaternion_to_yaw(df)
        df = get_direction_angle(df)
        # check yaw
        df, c_yaw = check_yaw(df, args.diff_yaw_in_deg)
        container.append({
            Columns.FILE_PATH: csv_path,
            Columns.YAW_PRECISION_RATIO: c_yaw * 100,
            Columns.DIFF_YAW_MAX: np.rad2deg(df['diff_yaw'].max()),
            Columns.DIFF_YAW_MEAN: np.rad2deg(df['diff_yaw'].mean()),
            Columns.DIFF_YAW_STD: np.rad2deg(df['diff_yaw'].std())
        })

    report = pd.DataFrame(container).sort_values(by=[Columns.YAW_PRECISION_RATIO],
                                                 ascending=False).reset_index(drop=True)
    print(report.to_string())
    # report.to_csv("report.csv", index=False)


if __name__ == "__main__":
    main(get_args())
