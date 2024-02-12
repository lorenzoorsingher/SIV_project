import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
from copy import copy

from kitti_loader import KittiLoader

ORB_BF = 0
ORB_FLANN = 1
SIFT_FLANN = 2
SIFT_KNN = 3

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

    return np.linalg.norm(gt_diff - agent_diff)


def update_map(pose, map, mapsize, color=(255, 255, 0)):

    # extract x, y, z from pose as well as rotation
    x, _, z = pose.T[-1]
    R = pose.T[:3].T

    # update trace of map
    map = cv.circle(
        map,
        (int(x) + mapsize, int(z) + mapsize),
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
            int(nx) + int(x) + mapsize,
            int(nz) + int(z) + mapsize,
        ),
        (int(x) + mapsize, int(z) + mapsize),
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


def draw_gt_map(map: np.ndarray, mapsize: int, kl: KittiLoader):
    for i in tqdm(range(0, len(kl.poses))):
        pose = kl.poses[i]
        x, _, z = pose.T[-1]

        # update trace of map
        map = cv.circle(
            map,
            (int(x) + mapsize, int(z) + mapsize),
            1,
            (255, 0, 100),
            4,
        )
    return map
