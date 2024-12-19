import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from typing import Tuple
import tqdm

class InterpolationBasedDetector:
    def __init__(self, window_size: int, threshold: float, step_size: int, order: int = 3):
        self.window_size = window_size
        self.threshold = threshold
        self.step_size = step_size
        self.order = order

    def interpolate(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        x = points[:, 0]
        y = points[:, 1]

        coefficients_y = np.polyfit(x, y, self.order)
        y_interpolated = np.polyval(coefficients_y, x)
        rmse_y = np.mean((y - y_interpolated)**2)

        coefficients_x = np.polyfit(y, x, self.order)
        x_interpolated = np.polyval(coefficients_x, y)
        rmse_x = np.mean((x - x_interpolated)**2)

        rmse = min(rmse_x, rmse_y)
        return coefficients_y if rmse_y < rmse_x else coefficients_x, rmse

    def curve_fitting_errors(self, trajectory: np.ndarray) -> np.ndarray:
        errors = []
        max_index = trajectory.shape[0] - self.window_size
        for i in tqdm.tqdm(range(0, max_index, self.step_size), leave=False):
            window = trajectory[i:i + self.window_size]
            _, error = self.interpolate(window)
            errors.append(error)
        return np.array(errors)

    def detect_discontinuities(self, trajectory: np.ndarray) -> np.ndarray:
        errors = self.curve_fitting_errors(trajectory)
        above_threshold = np.where(errors > self.threshold)[0]

        discontinuities = np.zeros(trajectory.shape[0])
        for idx in above_threshold:
            start = idx * self.step_size
            end = start + self.window_size
            discontinuities[start:end] = 1

        return discontinuities

    def read_discontinuities(self, discontinuities: np.ndarray) -> str:
        starts = np.where((discontinuities[1:] == 1) & (discontinuities[:-1] == 0))[0] + 1
        ends = np.where((discontinuities[:-1] == 1) & (discontinuities[1:] == 0))[0] + 1

        intervals = list(zip(starts, ends))
        return f"Discontinuity detected at {intervals}"

    def plot_error(self, trajectory: np.ndarray) -> None:
        errors = self.curve_fitting_errors(trajectory)
        plt.plot(errors)
        plt.title("Curve Fitting Errors")
        plt.xlabel("Window Index")
        plt.ylabel("Error")
        plt.show()

    def plot_trajectory_error(self, trajectory: np.ndarray, point_size: int = 1, title: str = None, colorbar: bool = True, cmap: str = 'inferno') -> None:
        errors = self.curve_fitting_errors(trajectory)
        plt.scatter(trajectory[:-self.window_size, 0], trajectory[:-self.window_size, 1], c=errors, s=point_size, cmap=cmap)
        if colorbar:
            plt.colorbar(label="Error")
        if title:
            plt.title(title, fontsize=10)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

class FilteringBasedDetector:
    @staticmethod
    def median_filter(data: np.ndarray, window_size: int) -> np.ndarray:
        return savgol_filter(data, window_size, 1)

    @staticmethod
    def mean_filter(signal: np.ndarray, window_size: int) -> np.ndarray:
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def savitzky_golay_filter(data: np.ndarray, window_size: int, order: int) -> np.ndarray:
        return savgol_filter(data, window_size, order)

    @staticmethod
    def rmse(traj1: np.ndarray, traj2: np.ndarray) -> float:
        return np.mean((traj1 - traj2) ** 2)
