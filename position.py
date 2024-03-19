import numpy as np

from common import *


class Position:
    """
    This class is used to keep track of the agent position,
    it contains the functionalities to update the location based
    on the rotation and translation between two consecutive frames.
    """

    def __init__(self):

        self.world_pose = np.eye(4, 4)
        self.cumul_R = np.eye(3, 3, dtype=np.float64)
        self.cumul_t = np.array([0, 0, 0], dtype=np.float64)
        self.lastgoodpose = np.eye(4, 4)

    def update_pos(self, R, t, bad_data, abs_scale) -> np.ndarray:
        """
        Update the agent position based on the rotation and translation

        Parameters
        ----------
        R (ndarray): Rotation matrix
        t (ndarray): Translation vector
        bad_data (bool): flag to indicate if the data is bad

        Returns
        -------
        ndarray: The updated agent world position
        """

        # Check if the data is bad. If data is bad
        # then use the last good pose, otherwise update
        # the last good pose with the current pose

        if not bad_data:
            heading = rotationMatrixToEulerAngles(R) * 180 / np.pi

            # TODO: there must be a better wahy to do this
            if abs(heading[1]) >= 4:
                bad_data = True
            #     print("X" * int(heading[1]))
            # else:
            #     print("#" * int(heading[1]))
        if bad_data:
            R = self.lastgoodpose[:3, :3]
            t = self.lastgoodpose[:3, 3]
            sizestr = ""
            sizestr2 = ""
            for i in range(len(sizestr)):
                sizestr2 += "X"
            sizestr = sizestr2
        else:
            self.lastgoodpose = np.eye(4, 4)
            self.lastgoodpose[:3, :3] = R
            self.lastgoodpose[:3, 3] = t

        # also adjusts t according to absolute scale
        self.cumul_t = abs_scale * t + np.dot(R, self.cumul_t)

        self.cumul_R = np.dot(R, self.cumul_R)

        # invert the coordinates system from camera to world
        # with this formula v = R^T * v' - R^T * t
        self.world_pose[:3, :3] = self.cumul_R.T
        self.world_pose[:3, 3] = np.dot(-(self.cumul_R).T, self.cumul_t)

        return self.world_pose
