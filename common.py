import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
from copy import copy

from kitti_loader import KittiLoader

# consts for feature matching
# as well as reverse lookup
ORB_BF_SORT = 0
ORB_FLANN_LOWE = 1
SIFT_FLANN_SORT = 2
SIFT_BF_LOWE = 3
ORB_BF_LOWE = 4
SIFT_FLANN_LOWE = 5
FM = [
    "ORB_BF_SORT",
    "ORB_FLANN_LOWE",
    "SIFT_FLANN_SORT",
    "SIFT_BF_LOWE",
    "ORB_BF_LOWE",
    "SIFT_FLANN_LOWE",
]

# consts for FLANN
FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_SAVED = 254
FLANN_INDEX_AUTOTUNED = 255

IMG_DEBUG = True


def rotationMatrixToEulerAngles(R) -> np.ndarray:
    """
    Convert a rotation matrix to Euler angles.

    Parameters
    ----------
    - R (ndarray): Rotation matrix

    Returns
    -------
    - ndarray: Euler angles
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def update_map(pose, map, origin, color=(255, 255, 0)):
    """
    Update the map with the current pose

    Parameters
    ----------
    - pose (ndarray): Current pose
    - map (ndarray): Map to update
    - origin (tuple): Origin of the map
    - color (tuple): Color of the trace

    Returns
    -------
    - ndarray: Updated map
    """

    # extract x and z from pose as well as rotation
    x, _, z = pose.T[-1]
    R = pose.T[:3].T

    # update trace of map
    map = cv.circle(
        map,
        (int(x) + origin[0], int(z) + origin[1]),
        1,
        color,
        2,
    )

    # copy map to avoid overwriting of direction arrow
    updated_map = copy(map)

    # create direction arrow pointing forward by 100 units
    forw = np.array([0, 0, 100, 1])

    newt = np.eye(4, 4)
    newt[:3, :3] = R
    newt[2, 3] = 1
    nx, _, nz, _ = newt @ forw

    # draw direction arrow
    updated_map = cv.line(
        updated_map,
        (
            int(nx) + int(x) + origin[0],
            int(nz) + int(z) + origin[1],
        ),
        (int(x) + origin[0], int(z) + origin[1]),
        (255, 0, 255),
        2,
    )

    # draw angle of rotation

    heading = rotationMatrixToEulerAngles(R) * 180 / np.pi

    cv.putText(
        updated_map,
        str(heading[1].round(2)),
        (10, updated_map.shape[0] // 10),
        cv.FONT_HERSHEY_PLAIN,
        updated_map.shape[0] // 100,
        (0, 0, 0),
        updated_map.shape[0] // 100,
        cv.LINE_AA,
    )
    return updated_map


def draw_gt_map(map: np.ndarray, origin: int, kl: KittiLoader):
    """
    Draw the ground truth map

    Parameters
    ----------
    - map (ndarray): Map to draw on
    - origin (int): Origin of the map
    - kl (KittiLoader): KittiLoader object

    Returns
    -------
    - ndarray: Updated map
    """

    for i in range(0, len(kl.poses)):
        pose = kl.poses[i]
        x, _, z = pose.T[-1]

        # update trace of map
        map = cv.circle(
            map,
            (int(x) + origin[0], int(z) + origin[1]),
            1,
            (255, 0, 100),
            4,
        )
    return map


def draw_maps(all_poses, no_gt=False):
    """
    Draw the maps for all the runs in all_poses

    Parameters
    ----------
    - all_poses (list): List of lists of poses (one for each run)
    - no_gt (bool): Flag to indicate if ground truth is not available

    Returns
    -------
    - ndarray: Updated map
    """

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green (Dark)
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]

    # computes map extremes and prepare canvas

    max_x = 0
    min_x = np.inf
    max_z = 0
    min_z = np.inf

    for poses in all_poses:

        max_x = max(max_x, int(np.array(poses)[:, 3].max()))
        min_x = min(min_x, int(np.array(poses)[:, 3].min()))
        max_z = max(max_z, int(np.array(poses)[:, 11].max()))
        min_z = min(min_z, int(np.array(poses)[:, 11].min()))

    margin = 50
    max_x += margin
    min_x -= margin
    max_z += margin
    min_z -= margin

    size_z = max_z - min_z
    size_x = max_x - min_x
    map_size = (size_z, size_x, 3)
    origin = (-min_x, -min_z)
    map = np.full(map_size, 255, dtype=np.uint8)

    # draw starting point
    x = all_poses[0][0][3]
    z = all_poses[0][0][11]
    map = cv.circle(
        map,
        (int(x) + origin[0], int(z) + origin[1]),
        1,
        (0, 0, 0),
        (size_z * size_x) // 20000,
    )

    # draw all poses
    for idx, poses in enumerate(all_poses):
        for pose in poses:
            x = pose[3]
            z = pose[11]

            if not no_gt and idx == 0:
                color = (0, 0, 0)
                linesize = (size_z * size_x) // 80000
            else:
                color = colors[idx]
                linesize = (size_z * size_x) // 130000
            map = cv.circle(
                map,
                (int(x) + origin[0], int(z) + origin[1]),
                1,
                color,
                linesize,
            )
    return map


def get_color(err, range=(0, 1)):
    """
    Get color based on error value, ranging between green and red

    Parameters
    ----------
    - err (float): Error value

    Returns
    -------
    - tuple: BGR color
    """
    B = 0
    G = min(int(255 - (err - range[0]) / (range[1] - range[0]) * 255), 255)
    R = int(255 - G)
    return (B, G, R)
