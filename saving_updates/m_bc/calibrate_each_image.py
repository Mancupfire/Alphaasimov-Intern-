import yaml
from yaml.loader import SafeLoader
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Calibrate raw image on alaska01')
parser.add_argument('-c', '--camera', dest='cam_name', default="head_camera", type=str, help='camera_name')
parser.add_argument('-m', '--matrix', dest='calib_matrix', default="head_camera.yaml", type=str, help='calibration matrix')
parser.add_argument('-i', '--input_dir', dest='input_folder',
                    default="/media/asimovsimpc/bulldog/aa-data/extracted_data/umtn/phenikaa/2.2/2023_12_23/2023_12_23_09_03_47/camera_front",
                    type=str, help='input folder which contains raw images')
parser.add_argument('-iw', '--width', dest='width', default=1280, type=int, help='image width')
parser.add_argument('-ih', '--height', dest='height', default=720, type=int, help='image height')

args = parser.parse_args()

cam_name = args.cam_name
calib_matrix = args.calib_matrix
input_folder = args.input_folder

# Create output folder if it doesn't exist
output_folder = f"{input_folder}_calibrated"
os.makedirs(output_folder, exist_ok=True)

# Open the file and load the file
with open(calib_matrix) as f:
    data = yaml.load(f, Loader=SafeLoader)
    print(data)

    matrix = np.array(data["camera_matrix"]["data"]).reshape((3, 3))
    distortion = np.array(data["distortion_coefficients"]["data"]).reshape((1, 5))

    print("Camera Matrix:")
    print(matrix)
    print("Distortion Coefficients:")
    print(distortion)

# Iterate through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        frame = cv2.imread(image_path)

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))

        # Method 1 to undistort the image
        calibrated_frame = cv2.undistort(frame, matrix, distortion, None, newcameramtx)
        x, y, w, h = roi
        calibrated_frame = calibrated_frame[y:y + h, x:x + w]

        calibrated_frame = cv2.resize(calibrated_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        # Save the calibrated image to the output folder
        # output_path = os.path.join(output_folder, f"undistorted_{filename}")
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, calibrated_frame)

print("Calibration complete. Calibrated images saved in:", output_folder)
