import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from typing import Tuple
import tqdm

class InterpolationBasedDetector:
    def __init__(self, window_size: int, threshold: float, step_size: int, order: int = 3):
        """
        Initialize the detector with parameters.

        Args:
            window_size (int): Size of the sliding window.
            threshold (float): Error threshold to detect discontinuities.
            step_size (int): Step size for the sliding window.
            order (int): Order of the polynomial for curve fitting (default is 3).
        """
        self.window_size = window_size
        self.threshold = threshold
        self.step_size = step_size
        self.order = order

    def interpolate(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform bidirectional curve fitting (y = f(x) and x = f(y)) to interpolate the points.

        Args:
            points (np.ndarray): Array of shape (n, 2) containing the points to interpolate.

        Returns:
            Tuple[np.ndarray, float]: Best-fitting polynomial coefficients and RMSE of the fit.
        """
        x = points[:, 0]
        y = points[:, 1]

        # Fit y = f(x)
        coefficients_y = np.polyfit(x, y, self.order)
        y_interpolated = np.polyval(coefficients_y, x)
        rmse_y = np.sqrt(np.mean((y - y_interpolated)**2))

        # Fit x = f(y)
        coefficients_x = np.polyfit(y, x, self.order)
        x_interpolated = np.polyval(coefficients_x, y)
        rmse_x = np.sqrt(np.mean((x - x_interpolated)**2))

        # Return the fit with the lower RMSE
        if rmse_y < rmse_x:
            return coefficients_y, rmse_y
        else:
            return coefficients_x, rmse_x

    def curve_fitting_errors(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute curve fitting errors for a trajectory using a sliding window.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.

        Returns:
            np.ndarray: Array of fitting errors for each window.
        """
        errors = []
        max_index = trajectory.shape[0] - self.window_size
        for i in tqdm.tqdm(range(0, max_index, self.step_size), leave=False):
            window = trajectory[i:i + self.window_size]
            _, error = self.interpolate(window)
            errors.append(error)
        return np.array(errors)

    def detect_discontinuities(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Detect discontinuities in the trajectory based on curve fitting errors.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.

        Returns:
            np.ndarray: Array indicating the detected discontinuities (1 for discontinuity, 0 otherwise).
        """
        errors = self.curve_fitting_errors(trajectory)
        above_threshold = np.where(errors > self.threshold)[0]

        # Mark discontinuity regions based on the threshold
        discontinuities = np.zeros(trajectory.shape[0])
        for idx in above_threshold:
            start = idx * self.step_size
            end = start + self.window_size
            discontinuities[start:end] = 1

        return discontinuities

    def read_discontinuities(self, discontinuities: np.ndarray) -> str:
        """
        Interpret and format detected discontinuities into readable intervals.

        Args:
            discontinuities (np.ndarray): Array of detected discontinuities.

        Returns:
            str: Message describing the intervals of detected discontinuities.
        """
        starts = np.where((discontinuities[1:] == 1) & (discontinuities[:-1] == 0))[0] + 1
        ends = np.where((discontinuities[:-1] == 1) & (discontinuities[1:] == 0))[0] + 1

        intervals = list(zip(starts, ends))
        return f"Discontinuity detected at {intervals}"

    def plot_error(self, trajectory: np.ndarray) -> None:
        """
        Plot the curve fitting errors for the given trajectory.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.
        """
        errors = self.curve_fitting_errors(trajectory)
        plt.plot(errors)
        plt.title("Curve Fitting Errors")
        plt.xlabel("Window Index")
        plt.ylabel("Error")
        plt.show()

    def plot_trajectory_error(self, trajectory: np.ndarray, point_size: int = 1, title: str = None, colorbar: bool = True, cmap: str = 'inferno') -> None:
        """
        Visualize the trajectory errors as a scatter plot.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.
            point_size (int): Size of points in the scatter plot.
            title (str): Title of the plot (optional).
            colorbar (bool): Whether to include a colorbar.
            cmap (str): Colormap for the scatter plot.
        """
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
        """
        Apply a median filter to the data.

        Args:
            data (np.ndarray): Array of data to filter.
            window_size (int): Size of the filtering window.

        Returns:
            np.ndarray: Filtered data.
        """
        return savgol_filter(data, window_size, 1)

    @staticmethod
    def mean_filter(signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply a mean filter to the data.

        Args:
            signal (np.ndarray): Input signal to filter.
            window_size (int): Size of the filtering window.

        Returns:
            np.ndarray: Filtered signal.
        """
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def savitzky_golay_filter(data: np.ndarray, window_size: int, order: int) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth the data.

        Args:
            data (np.ndarray): Input data to smooth.
            window_size (int): Size of the smoothing window.
            order (int): Polynomial order of the filter.

        Returns:
            np.ndarray: Smoothed data.
        """
        return savgol_filter(data, window_size, order)

    @staticmethod
    def rmse(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compute the Root Mean Square Error (RMSE) between two trajectories.

        Args:
            traj1 (np.ndarray): First trajectory (n, 2).
            traj2 (np.ndarray): Second trajectory (n, 2).

        Returns:
            float: RMSE value.
        """
        return np.sqrt(np.mean((traj1 - traj2) ** 2))
