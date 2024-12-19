import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import savgol_filter
from typing import Tuple


class InterpolationBasedDetector:
    def __init__(self, window_size: int, threshold: float, step_size: int, order: int = 3):
        """
        Initialize the detector with parameters for curve fitting.

        Args:
            window_size (int): Size of the sliding window.
            threshold (float): Error threshold for discontinuity detection.
            step_size (int): Step size for sliding window.
            order (int): Polynomial order for curve fitting (default: 3).
        """
        self.window_size = window_size
        self.threshold = threshold
        self.step_size = step_size
        self.order = order

    def interpolate(self, points: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Perform bidirectional polynomial fitting and select the better fit.

        Args:
            points (np.ndarray): Array of shape (n, 2) representing points to fit.

        Returns:
            Tuple[int, np.ndarray, float]: Fit type (0 for x=f(y), 1 for y=f(x)),
                                           coefficients, and RMSE for the better fit.
        """
        x, y = points[:, 0], points[:, 1]

        # Fit y = f(x)
        coeff_y = np.polyfit(x, y, self.order)
        rmse_y = np.sqrt(np.mean((y - np.polyval(coeff_y, x)) ** 2))

        # Fit x = f(y)
        coeff_x = np.polyfit(y, x, self.order)
        rmse_x = np.sqrt(np.mean((x - np.polyval(coeff_x, y)) ** 2))

        if rmse_x < rmse_y:
            return 0, coeff_x, rmse_x  # x = f(y)
        else:
            return 1, coeff_y, rmse_y  # y = f(x)

    def plot_fitting_curve(self, trajectory: np.ndarray, index: int, ax: plt.Axes = None,
                           rmse=None, coefficients=None, fit_type_id=None) -> None:
        """
        Visualize the fitting curve at a specific trajectory segment.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.
            index (int): Index of the sliding window to plot.
            ax (plt.Axes, optional): Axes to plot on (creates new if None).
            rmse (float, optional): RMSE of the fit.
            coefficients (np.ndarray, optional): Polynomial coefficients.
            fit_type_id (int, optional): Fit type (0 for x=f(y), 1 for y=f(x)).
        """
        start, end = index * self.step_size, index * self.step_size + self.window_size
        window = trajectory[start:end]

        if coefficients is None or fit_type_id is None:
            fit_type_id, coefficients, _ = self.interpolate(window)

        # Generate points for plotting
        if fit_type_id == 0:  # x = f(y)
            y_fit = np.linspace(window[:, 1].min(), window[:, 1].max(), 100)
            x_fit = np.polyval(coefficients, y_fit)
        else:  # y = f(x)
            x_fit = np.linspace(window[:, 0].min(), window[:, 0].max(), 100)
            y_fit = np.polyval(coefficients, x_fit)

        # Plot data and the fitting curve
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(window[:, 0], window[:, 1], label="Original Points", zorder=1)
        ax.plot(x_fit, y_fit, color="red", label="Fitted Curve", zorder=2)

        ax.set_title(f"Fitting Curve at Index {index}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.show()

    def curve_fitting_errors(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute RMSE for each sliding window along the trajectory.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.

        Returns:
            np.ndarray: Array of RMSE values for each window.
        """
        errors = []
        max_index = trajectory.shape[0] - self.window_size
        for i in tqdm.tqdm(range(0, max_index, self.step_size), leave=False):
            window = trajectory[i:i + self.window_size]
            _, _, error = self.interpolate(window)
            errors.append(error)
        return np.array(errors)

    def detect_discontinuities(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Detect discontinuities based on curve fitting errors.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.

        Returns:
            np.ndarray: Binary array indicating discontinuity (1 for discontinuity, 0 otherwise).
        """
        errors = self.curve_fitting_errors(trajectory)
        discontinuities = np.zeros(trajectory.shape[0])
        high_error_indices = np.where(errors > self.threshold)[0]

        for i in high_error_indices:
            start, end = i * self.step_size, i * self.step_size + self.window_size
            discontinuities[start:end] = 1
        return discontinuities

    def read_discontinuities(self, discontinuities: np.ndarray) -> str:
        """
        Convert detected discontinuities into human-readable intervals.

        Args:
            discontinuities (np.ndarray): Binary array indicating discontinuities.

        Returns:
            str: Human-readable discontinuity intervals.
        """
        starts = np.where((discontinuities[1:] == 1) & (discontinuities[:-1] == 0))[0] + 1
        ends = np.where((discontinuities[:-1] == 1) & (discontinuities[1:] == 0))[0] + 1
        intervals = list(zip(starts, ends))
        return f"Discontinuities detected at intervals: {intervals}"

    def plot_error(self, trajectory: np.ndarray) -> None:
        """
        Plot the curve fitting error for the trajectory.

        Args:
            trajectory (np.ndarray): Array of shape (n, 2) representing the trajectory.
        """
        errors = self.curve_fitting_errors(trajectory)
        plt.plot(errors)
        plt.title("Curve Fitting Errors")
        plt.xlabel("Window Index")
        plt.ylabel("Error")
        plt.show()


class FilteringBasedDetector:
    @staticmethod
    def median_filter(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply a median filter to smooth the data.

        Args:
            data (np.ndarray): Input data.
            window_size (int): Size of the window.

        Returns:
            np.ndarray: Smoothed data.
        """
        return savgol_filter(data, window_size, 1)

    @staticmethod
    def mean_filter(signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply a mean filter to the data.

        Args:
            signal (np.ndarray): Input signal.
            window_size (int): Size of the window.

        Returns:
            np.ndarray: Smoothed signal.
        """
        return np.convolve(signal, np.ones(window_size) / window_size, mode="same")

    @staticmethod
    def savitzky_golay_filter(data: np.ndarray, window_size: int, order: int) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth the data.

        Args:
            data (np.ndarray): Input data.
            window_size (int): Size of the smoothing window.
            order (int): Polynomial order.

        Returns:
            np.ndarray: Smoothed data.
        """
        return savgol_filter(data, window_size, order)


# Example Usage
if __name__ == "__main__":
    # Example usage of InterpolationBasedDetector
    detector = InterpolationBasedDetector(window_size=50, threshold=0.5, step_size=10, order=3)
    trajectory = np.random.rand(1000, 2)  # Example trajectory
    discontinuities = detector.detect_discontinuities(trajectory)
    print(detector.read_discontinuities(discontinuities))
    detector.plot_error(trajectory)
