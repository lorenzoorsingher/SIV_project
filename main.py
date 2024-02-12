import os
import json
from copy import copy

import cv2 as cv
import numpy as np
from tqdm import tqdm
from common import *
from odometry import VOAgent
from kitti_loader import kittiLoader


imgs_path = "data/sequence_14/images"

calib_path = "camera_data/calib.json"
maxdist = 300

cv.namedWindow("gt_map", cv.WINDOW_NORMAL)
# cv.namedWindow("frames", cv.WINDOW_NORMAL)

np.set_printoptions(formatter={"all": lambda x: str(x)})

DEBUG = True
STEPS = 100
FRAMESKIP = 1
MODE = "kitti"
if MODE == "imgs":
    calib_path = "camera_data/calib_tum.json"
    im_paths = [imgs_path + "/" + name for name in os.listdir(imgs_path)]
    im_paths.sort()
    idx = 0

    data = json.load(open(calib_path))
    mtx = np.array(data[0])
    dist = np.array(data[1])
if MODE == "video":
    calib_path = "camera_data/calib.json"
    video_path = "data/room_tour.MOV"
    cap = cv.VideoCapture(video_path)

    data = json.load(open(calib_path))
    mtx = np.array(data[0])
    dist = np.array(data[1])
if MODE == "kitti":
    do_images = "data/data_odometry_gray/dataset/sequences"
    do_poses = "data/data_odometry_poses/dataset/poses"

    SEQUENCE = 0
    kl = kittiLoader(do_images, do_poses, SEQUENCE)
    mtx, dist = kl.get_params()
    maxdist = int(kl.get_maxdist())
    STEPS = kl.get_seqlen()


def update_map(pose, map, color=(255, 255, 0)):

    # extract x, y, z from pose as well as rotation
    x, _, z = pose.T[-1]
    R = pose.T[:3].T

    # update trace of map
    map = cv.circle(
        map,
        (int(x) + maxdist, int(z) + maxdist),
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
            int(nx) + int(x) + maxdist,
            int(nz) + int(z) + maxdist,
        ),
        (int(x) + maxdist, int(z) + maxdist),
        (255, 0, 255),
        2,
    )

    # draw angle of rotation
    cv.putText(
        updated_map,
        str(odo.position.heading[1].round(2)),
        (10, updated_map.shape[0] // 10),
        cv.FONT_HERSHEY_PLAIN,
        updated_map.shape[0] // 100,
        (255, 255, 255),
        updated_map.shape[0] // 100,
        cv.LINE_AA,
    )
    return updated_map


def draw_gt_map(map):
    for i in range(0, len(kl.poses)):
        pose = kl.poses[i]
        x, _, z = pose.T[-1]

        # update trace of map
        map = cv.circle(
            map,
            (int(x) + maxdist, int(z) + maxdist),
            1,
            (0, 50, 50),
            1,
        )
    return map


def compute_absolute_scale(old_pose, new_pose):
    """
    Computes the absolute scale between two poses
    """
    return np.linalg.norm(old_pose[:, 3] - new_pose[:, 3])


odo = VOAgent(mtx, dist, buf_size=1, matcher_method=SIFT_KNN)


maxdist = int(maxdist * 1.5)
mapsize = maxdist * 2 + 10

gt_map = np.zeros((mapsize, mapsize, 3), dtype=np.uint8)
track_map = np.zeros((mapsize, mapsize, 3), dtype=np.uint8)
updated_gt_map = np.zeros((mapsize, mapsize, 3), dtype=np.uint8)

old_gt_pose = np.eye(3, 4, dtype=np.float64)
old_agent_pose = np.eye(3, 4, dtype=np.float64)


errors = []

for tqdm_idx in range(STEPS):
    if MODE == "video":
        for i in range(FRAMESKIP):
            ret, frame = cap.read()
    if MODE == "imgs":
        for i in range(FRAMESKIP):
            frame = cv.imread(im_paths[idx])
            idx += 1
        ret = True
    if MODE == "kitti":
        for i in range(FRAMESKIP):
            frame, gt_pose = kl.next_frame()
        updated_gt_map = update_map(gt_pose, gt_map)

        ret = True

    if not ret:
        break

    abs_scale = np.linalg.norm(old_gt_pose[:, 3] - gt_pose[:, 3])
    # print(abs_scale)

    agent_pose = odo.next_frame(frame, abs_scale)[:-1]

    # breakpoint()
    err = eval_error(old_gt_pose, gt_pose, old_agent_pose, agent_pose)
    color = (0, int(min(255, (1 - err * 1.4) * 255)), int(min(255, (err * 1.4) * 255)))
    # color = (0, int(min(255, (1 - err * 1.4) * 255)), 0)
    old_gt_pose = gt_pose.copy()
    old_agent_pose = agent_pose.copy()

    errors.append(err)

    # TODO: add error evaluation
    # TODO: add performance evaluation
    # TODO: remember in a monocular system we can only estimate t up to a scale factor
    if DEBUG:
        updated_track_map = update_map(agent_pose, track_map, color)
        cv.imshow("gt_map", np.hstack([updated_gt_map, updated_track_map]))
        cv.waitKey(1)

print("avg: ", np.mean(errors).round(3), " max: ", np.max(errors).round(3))
breakpoint()
