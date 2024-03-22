import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
from copy import copy

from kitti_loader import KittiLoader

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
    R (ndarray): Rotation matrix

    Returns
    -------
    ndarray: Euler angles
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


def eval_error(old_gt_pose, gt_pose, old_agent_pose, agent_pose):

    gt_diff = (old_gt_pose - gt_pose)[:, 3]
    agent_diff = (old_agent_pose - agent_pose)[:, 3]

    Ra = agent_pose[:, :3]
    Rg = gt_pose[:, :3]

    new_a = Ra.T @ agent_diff
    new_g = Rg.T @ gt_diff
    new_err = np.linalg.norm(new_g - new_a)

    err = np.linalg.norm(gt_diff - agent_diff)

    return new_err


def update_map(pose, map, origin, color=(255, 255, 0)):

    # extract x, y, z from pose as well as rotation
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


def draw_maps(all_poses):

    colors = [
        (0, 0, 0),
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

    max_x = 0
    min_x = np.inf
    max_z = 0
    min_z = np.inf

    for poses in all_poses:

        max_x = max(max_x, int(np.array(poses)[:, 3].max()))
        min_x = min(min_x, int(np.array(poses)[:, 3].min()))
        max_z = max(max_z, int(np.array(poses)[:, 11].max()))
        min_z = min(min_x, int(np.array(poses)[:, 11].min()))
    # create and prepare maps
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

    x = all_poses[0][0][3]
    z = all_poses[0][0][11]
    map = cv.circle(
        map,
        (int(x) + origin[0], int(z) + origin[1]),
        1,
        (0, 0, 0),
        (size_z * size_x) // 20000,
    )

    for idx, poses in enumerate(all_poses):
        for pose in poses:
            x = pose[3]
            z = pose[11]
            # update trace of map
            if idx == 0:
                linesize = (size_z * size_x) // 80000
            else:
                linesize = (size_z * size_x) // 130000
            map = cv.circle(
                map,
                (int(x) + origin[0], int(z) + origin[1]),
                1,
                colors[idx],
                linesize,
            )
    return map


def get_color(err, range=(0, 1)):
    """
    Get color based on error value
    """
    B = 0
    G = min(int(255 - (err - range[0]) / (range[1] - range[0]) * 255), 255)
    R = int(255 - G)
    return (B, G, R)
