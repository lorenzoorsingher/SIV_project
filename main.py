import os
import json
from copy import copy

import cv2 as cv
import numpy as np
from tqdm import tqdm

from common import *
from VOAgent import VOAgent
from kitti_loader import KittiLoader


cv.namedWindow("gt_map", cv.WINDOW_NORMAL)
# cv.namedWindow("frames", cv.WINDOW_NORMAL)

np.set_printoptions(formatter={"all": lambda x: str(x)})

DEBUG = True
STEPS = 100
FRAMESKIP = 1
ORIGIN_COO = 300
MODE = "kitti"

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

    SEQUENCE = 22
    kl = KittiLoader(do_images, do_poses, SEQUENCE)
    mtx, dist = kl.get_params()
    maxdist = int(kl.get_maxdist())
    kl.set_idx(0)
    STEPS = kl.get_seqlen()
    ORIGIN_COO = int(maxdist * 1.5)

# create Visual Odometry Agent
odo = VOAgent(mtx, dist, buf_size=1, matcher_method=SIFT_FLANN_LOWE)

# create and prepare maps
(max_x, min_x, max_z, min_z) = kl.get_extremes()
margin = 50
max_x += margin
min_x -= margin
max_z += margin
min_z -= margin

size_z = max_z - min_z
size_x = max_x - min_x
map_size = (size_z, size_x, 3)
origin = (-min_x, -min_z)

# gt_map = np.full((mapsize, mapsize, 3), 255, dtype=np.uint8)
track_map = np.full(map_size, 255, dtype=np.uint8)
updated_gt_map = np.full(map_size, 255, dtype=np.uint8)
track_map = draw_gt_map(track_map, origin, kl)

# initialize poses
old_gt_pose = np.eye(3, 4, dtype=np.float64)
old_agent_pose = np.eye(3, 4, dtype=np.float64)
errors = []

abs_scale = 1
for tqdm_idx in range(STEPS):

    if MODE == "video":
        for i in range(FRAMESKIP):
            ret, frame = cap.read()
            if not ret:
                break

    if MODE == "kitti":
        for i in range(FRAMESKIP):
            frame, gt_pose = kl.next_frame()
        # updated_gt_map = update_map(gt_pose, gt_map)
        # compute absolute scale from ground truth
        abs_scale = np.linalg.norm(old_gt_pose[:, 3] - gt_pose[:, 3])

    agent_pose = odo.next_frame(frame, abs_scale)[:-1]

    # error evaluation
    err = eval_error(old_gt_pose, gt_pose, old_agent_pose, agent_pose)

    old_gt_pose = gt_pose.copy()
    old_agent_pose = agent_pose.copy()

    errors.append(err)

    # TODO: make error evalluation rotation invariant
    # TODO: add performance evaluation
    # TODO: remember in a monocular system we can only estimate t up to a scale factor
    if DEBUG:
        color = (0, 200, 0)
        color = get_color(err, range=(0, 0.5))
        # print("error: ", err)
        updated_track_map = update_map(agent_pose, track_map, origin, color)
        # cv.imshow("gt_map", np.hstack([updated_gt_map, updated_track_map]))
        cv.imshow("gt_map", updated_track_map)
        cv.imwrite("output/map_" + str(tqdm_idx) + ".png", updated_track_map)
        cv.waitKey(1)

print("avg: ", np.mean(errors).round(3), " max: ", np.max(errors).round(3))
breakpoint()
