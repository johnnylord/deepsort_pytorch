import numpy as np
import scipy.linalg


class KalmanFilter:
    """Kalman filter for predicting bounding boxes in image space

    Here are the targets we are trying to predict

        [ x, y, a, h, vx, vy, va, vh ]

    Target explanation:
        - x: bounding box center postion of x dimension
        - y: bounding box center postion of y dimension
        - a: aspect ratio of bounding box width and bounding box height
        - h: bounding box height
        - v*: respective velocities of (x, y, a, h)

    The motion model is a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation.
    """
    def __init__(self):
        """
        Attributes:
        - _motion_mat
            transform matrix for predicting next state of targets
        - _project_mat
            transform matrix for projecting mean vector (8 dimensional) to the
            measurment/observation space, which is (4 dimensional).
        - _std_position
            standard deviation of the position estimation/measurement (x, y, a, h)
        - _std_velocity
            standard deviation of the velocity estimation/measurement (vx, vy, va, vh)
        """
        n_dim = 4
        delta_time = 1.

        # Dimension of motion matrix: (8, 8)
        # Dimension of mask matrix: (4, 8)
        self._motion_mat = np.eye(2*n_dim, 2*n_dim)
        for i in range(n_dim):
            self._motion_mat[i, n_dim+i] = delta_time
        self._project_mat = np.eye(n_dim, 2*n_dim)

        # Noise/Error in the prediction
        self._std_position = 1. / 20
        self._std_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement

        Parameters:
        - meansurement: ndarray
            bounding box coordinate (x, y, a, h) with center position (x,y)
            aspect ratio a, and height h

        Return:
        - (ndarray, ndarray)
            Return the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track.
        """
        # Mean vector
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        # Covariance matrix
        # The initial standard deviation matrix is affected by HEIGHT of bbox
        std = [
            2 * self._std_position * measurement[3],
            2 * self._std_position * measurement[3],
            1e-2,
            2 * self._std_position * measurement[3],
            10 * self._std_velocity * measurement[3],
            10 * self._std_velocity * measurement[3],
            1e-5,
            10 * self._std_velocity * measurement[3]]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step

        Paramters:
        - mean: ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        - covariance: ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Return:
        - (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted state
        """
        # Noise for covariance matrix
        std_position = [
            self._std_position * mean[3],
            self._std_position * mean[3],
            1e-2,
            self._std_position * mean[3]]
        std_velocity = [
            self._std_velocity * mean[3],
            self._std_velocity * mean[3],
            1e-5,
            self._std_velocity * mean[3]]
        motion_covariance = np.diag(np.square(np.concatenate([std_position, std_velocity])))

        # Update mean vector and covariance matrix
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_covariance

        return mean, covariance

    def _project(self, mean, covariance):
        """Project mean and covariance to observation/meansurement space

        Parameters:
        - mean: ndarray
            The predicted state's mean vector (8 dimensional)
        - covariance: ndarray
            the state's covariance matrix (8x8 covariance)

        Return:
        - (ndarray, ndarray)
            A projected mean in measurement space (4 dimensional), and a projected
            covariance matrix in measurement space (4x4 dimensional)
        """
        # Noise for projected covariance matrix
        std_position = [
            self._std_position * mean[3],
            self._std_position * mean[3],
            1e-1,
            self._std_position * mean[3]]
        projected_covariance = np.diag(np.square(std_position))

        # Projected mean and covariance matrix
        mean = np.dot(self._project_mat, mean)
        covariance = np.linalg.multi_dot((
            self._project_mat, covariance, self._project_mat.T)) + projected_covariance

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step

        Parameter:
        - mean: ndarray
            The predicted state's mean vector (8 dimensional).
        - covariacne: ndarray
            The state's covariance matrix (8x8 dimensional)
        - measurement: ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y) is
            the center position, a is the aspect ratio, h is height of the
            bounding box.

        Return:
        - (ndarray, ndarray)
            Returns the meansurement-corrected state distribution
        """
        # Project mean & covariance so that they are in the same space as measurement
        projected_mean, projected_covariance = self._project(mean, covariance)

        # Calculate kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
                                    projected_covariance,
                                    lower=True,
                                    check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
                                    (chol_factor, lower),
                                    np.dot(covariance, self._project_mat.T).T,
                                    check_finite=False).T

        # Update mean and covariance with measurement
        mean = mean + np.dot(measurement-projected_mean, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))

        return mean, covariance
