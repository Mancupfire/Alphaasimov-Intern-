import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import medfilt, savgol_filter
from scipy.optimize import minimize
from typing import Tuple

class InterpolationBasedDetector:
    def __init__(self, window_size: int, threshold: float, step_size: int, order: int = 3):
        self.window_size = window_size
        self.threshold = threshold
        self.step_size = step_size
        self.order = order

    def objective_function(self, coefficients, points):
            """
            Objective function for curve fitting

            Args:
                coefficients : np.ndarray
                    Coefficients of the polynomial, shape (2*order+2,)
                points : np.ndarray
                    Discrete points of shape (n, 2) (first column represent x, second column represent y)

            Returns:
                float
            """
            x = points[:, 0]
            y = points[:, 1]

            a = coefficients[:self.order+1]
            b = coefficients[self.order+1:]

            poly_x = np.polyval(a, x)
            poly_y = np.polyval(b, y)

            error = np.sum((poly_x + poly_y)**2)/len(x)
            return error

    def interpolate(self, points: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Interpolate continuous function from set of discrete points

        Args:
            points : np.ndarray
                Discrete points of shape (n, 2) (first column represent x, second column represent y)

        Returns:
            fit_type_id : int
                0 if x = f(y) is the better fit, 1 if y = f(x) is the better fit
            coefficients : np.ndarray
                Coefficients for the better fit (either x = f(y) or y = f(x))
            rmse : float
        """
        x = points[:, 0]
        y = points[:, 1]

        # Bidirectional curve fitting
        coefficients_y = np.polyfit(x, y, self.order)
        y_interpolated = np.polyval(coefficients_y, x)
        rmse_y = np.sqrt(np.mean((y - y_interpolated) ** 2)) 

        coefficients_x = np.polyfit(y, x, self.order)
        x_interpolated = np.polyval(coefficients_x, y)
        rmse_x = np.sqrt(np.mean((x - x_interpolated) ** 2))

        if rmse_x < rmse_y:
            fit_type_id = 0  # x = f(y) is the better fit
            coefficients = coefficients_x
            rmse = rmse_x
        else:
            fit_type_id = 1  # y = f(x) is the better fit
            coefficients = coefficients_y
            rmse = rmse_y

        return fit_type_id, coefficients, rmse

    def plot_fitting_curve(self, trajectory: np.ndarray, index: int, ax: plt.Axes = None, rmse=None, coefficients=None,
                           fit_type_id=None) -> None:
        """
        Visualize the fitting curve at a specific index of the trajectory

        Args:
            trajectory: np.ndarray
                Discrete points of shape (n, 2)
            index: int
                Index of the window to plot
            ax: plt.Axes (optional)
                Axes to plot on. If not provided, a new figure will be created.
            rmse: float (optional)
                RMSE value from the log
            coefficients: np.ndarray (optional)
                Coefficients from the log
            fit_type_id: int (optional)
                Fit type ID from the log (0 for x = f(y), 1 for y = f(x))
        """
        start = index * self.step_size
        end = start + self.window_size
        window = trajectory[start:end]

        if rmse is None or coefficients is None or fit_type_id is None:
            fit_type_id, coefficients, rmse = self.interpolate(window)

        # Generate points for plotting the fitted curve across the entire window
        if fit_type_id == 0:  # x = f(y)
            y_fit = np.linspace(window[:, 1].min(), window[:, 1].max(), 100)
            x_fit = np.polyval(coefficients, y_fit)
        else:  # fit_type_id == 1 (y = f(x))
            x_fit = np.linspace(window[:, 0].min(), window[:, 0].max(), 100)
            y_fit = np.polyval(coefficients, x_fit)

        # Calculate error at each point on the curve
        if fit_type_id == 0:
            y_original = np.interp(x_fit, window[:, 0], window[:, 1])
            errors = np.abs(y_original - y_fit)
        else:
            x_original = np.interp(y_fit, window[:, 1], window[:, 0])
            errors = np.abs(x_original - x_fit)

        # Plot the original points and the fitted curve with colormap representing error
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(window[:, 0], window[:, 1], label='Original Points', zorder=1)
        scatter = ax.scatter(x_fit, y_fit, c=errors, cmap='viridis', label='Fitted Curve with Errors', zorder=2)
        plt.colorbar(scatter)

        ax.set_title(f'Fitted Curve at Index {index}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        # Explicitly set axis limits based on the window data
        ax.set_xlim(window[:, 0].min(), window[:, 0].max())
        ax.set_ylim(window[:, 1].min(), window[:, 1].max())

        if ax is None:
            plt.show()

    def rmse(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compute RMSE between 2 set of point.

        Args:
            traj1, traj2 : np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            float
        """
        error = np.sum((traj1 - traj2)**2)/len(traj1)
        return error

    def curve_fitting_errors(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply curve fitting to the trajectory using sliding window and compute the error

        Args:
            trajectory : np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            np.ndarray
        """
        errors = []
        max_index = trajectory.shape[0] - self.window_size
        for i in tqdm.tqdm(range(0, max_index, self.step_size), leave=False):
            window = trajectory[i:i+self.window_size]
            _,_, error = self.interpolate(window)
            errors.append(error)

        return np.array(errors)

    def detect_discontinuities(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Detect discontinuity on the trajectory

        Args:
            trajectory : np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            np.ndarray
        """
        errors = self.curve_fitting_errors(trajectory)
        above_threshold = np.where(errors > self.threshold)[0]

        discontinuities = np.zeros(trajectory.shape[0])
        for i in range(0, len(above_threshold)):
            start = above_threshold[i] * self.step_size
            end = start + self.window_size
            discontinuities[start:end] = 1

        return discontinuities

    def read_discontinuities(self, discontinuities: np.ndarray) -> str:
        """
        Read the discontinuity on the trajectory

        Args:
            discontinuities : np.ndarray
                Discontinuity points of shape (n, 1)

        Returns:
            np.ndarray
        """
        starts = np.where((discontinuities[1:] == 1) & (discontinuities[:-1] == 0))[0] + 1
        ends = np.where((discontinuities[:-1] == 1) & (discontinuities[1:] == 0))[0] + 1

        intervals = list(zip(starts, ends))

        message = f"Discontinuity detected at {intervals}"
        return message


    def plot_error(self, trajectory: np.ndarray) -> None:
        """
        Visualize the error of the trajectory

        Args:
            trajectory : np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            np.ndarray
        """
        errors = self.curve_fitting_errors(trajectory)
        plt.plot(errors)

    # def plot_trajectory_error(
    #         self,
    #         trajectory: np.ndarray,
    #         point_size: int = 1,
    #         title: str = None,
    #         colorbar: bool = True,
    #         cmap: str = 'inferno',
    #         yaw: np.ndarray = None,
    #         ax: plt.Axes = None
    #     ) -> None:
    #     """
    #     Visualize the error on the robots trajectory

    #     Args:
    #         trajectory : np.ndarray
    #             Discrete points of shape (n, 2)
    #         point_size : int
    #             Size of the point
    #         title : str
    #             Title of the plot
    #         colorbar : bool
    #             Show colorbar
    #         cmap : str
    #             Colormap
    #         yaw : np.ndarray
    #             Yaw of the robot
    #         ax : plt.Axes
    #             Axes to plot

    #     Returns:
    #         np.ndarray
    #     """
    #     errors = self.curve_fitting_errors(trajectory)
    #     if ax is None:
    #         ax = plt.gca()

    #     if yaw is None:
    #         scatter = ax.scatter(trajectory[:-self.window_size, 0],
    #                     trajectory[:-self.window_size, 1],
    #                     c=errors, # need to consider the window size
    #                     s=point_size,
    #                     cmap=cmap
    #         )
    #     else:
    #         quiver = ax.quiver(trajectory[:-self.window_size, 0],
    #                     trajectory[:-self.window_size, 1],
    #                     np.cos(yaw[:-self.window_size]),
    #                     np.sin(yaw[:-self.window_size]),
    #                     errors,
    #                     angles='xy',
    #                     scale_units='xy',
    #                     scale=1,
    #                     cmap=cmap
    #         )


    #     if colorbar:
    #         plt.colorbar(scatter if yaw is None else quiver, ax=ax)
    #     if title:
    #         ax.set_title(title, fontsize=8)

    #     return ax

    def plot_trajectory_error(
            self,
            trajectory: np.ndarray,
            point_size: int = 1,
            title: str = None,
            colorbar: bool = True,
            cmap: str = 'inferno',
            yaw: np.ndarray = None,
            ax: plt.Axes = None  # Add ax as an optional argument
        ) -> None:
        """
        Visualize the error on the robots trajectory

        Args:
            trajectory : np.ndarray
                Discrete points of shape (n, 2)
            point_size : int
                Size of the point
            title : str
                Title of the plot
            colorbar : bool
                Show colorbar
            cmap : str
                Colormap
            yaw : np.ndarray
                Yaw of the robot

        Returns:
            np.ndarray
        """
        errors = self.curve_fitting_errors(trajectory)

        if ax is None:  # Create a new figure and axes if not provided
            fig, ax = plt.subplots()

        if yaw is None:
            scatter = ax.scatter(trajectory[:-self.window_size, 0],
                        trajectory[:-self.window_size, 1],
                        c=errors,
                        s=point_size,
                        cmap=cmap
                        )
        else:
            quiver = ax.quiver(trajectory[:-self.window_size, 0],
                        trajectory[:-self.window_size, 1],
                        np.cos(yaw[:-self.window_size]),
                        np.sin(yaw[:-self.window_size]),
                        errors,
                        angles='xy',
                        scale_units='xy',
                        scale=1,
                        cmap=cmap
                        )

        if colorbar:
            plt.colorbar(scatter if yaw is None else quiver, ax=ax) 
        if title:
            ax.set_title(title, fontsize=8)
    def euclidean_distances(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the Euclidean distances between consecutive points in the trajectory

        Args:
            trajectory: np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            np.ndarray: Euclidean distances between consecutive points
        """
        return np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)

    def detect_distance_anomalies(self, trajectory: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Detect points in the trajectory where the Euclidean distance between
        consecutive points exceeds the threshold

        Args:
            trajectory: np.ndarray
                Discrete points of shape (n, 2)
            threshold: float
                Threshold for Euclidean distance

        Returns:
            np.ndarray: Boolean array indicating anomaly points
        """
        distances = self.euclidean_distances(trajectory)
        anomalies = distances > threshold  # Directly compare distances to the threshold

        return anomalies


class GradientBasedDetector:
    pass

class FilteringBasedDetector:
    def __init__(self):
        ...

    def median_filter(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Apply median filter to the data

        Args:
            data : np.ndarray
                Shape (n, 1)
            window_size : int
                Size of the window

        Returns:
            np.ndarray
        """
        return np.median(data, window_size)

    def mean_filter(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

    def savitzky_golay_filter(
            self, data: np.ndarray, window_size: int, order: int
        ) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to the data

        Args:
            data : np.ndarray
                Shape (n, 1)
            window_size : int
                Size of the window
            order : int
                Order of the polynomial

        Returns:
            np.ndarray
        """
        return savgol_filter(data, window_size, order)

    def rmse(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compute RMSE between 2 set of point.

        Args:
            traj1, traj2 : np.ndarray
                Discrete points of shape (n, 2)

        Returns:
            float
        """
        error = np.sum((traj1 - traj2)**2)/len(traj1)
        return error


