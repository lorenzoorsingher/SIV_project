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
cv.namedWindow("frames", cv.WINDOW_NORMAL)

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


def update_map(pose, map):

    # extract x, y, z from pose as well as rotation
    x, _, z = pose.T[-1]
    R = pose.T[:3].T

    # update trace of map
    map = cv.circle(
        map,
        (int(x) + maxdist, int(z) + maxdist),
        1,
        (255, 255, 0),
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


odo = VOAgent(mtx, dist, buf_size=1, matcher_method=SIFT_KNN)


maxdist = int(maxdist * 1.5)
mapsize = maxdist * 2 + 10
gt_map = np.zeros((mapsize, mapsize, 3))
track_map = np.zeros((mapsize, mapsize, 3))
# track_map = draw_gt_map(track_map)
updated_gt_map = np.zeros((mapsize, mapsize, 3))

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
            frame, pose = kl.next_frame()
        updated_gt_map = update_map(pose, gt_map)
        set_idx = 20
        ret = True

    if not ret:
        break

    odo.next_frame(frame)

    # TODO: add error evaluation
    # TODO: add performance evaluation

    if DEBUG:
        updated_track_map = update_map(odo.position.world_pose[:-1], track_map)
        cv.imshow("gt_map", np.hstack([updated_gt_map, updated_track_map]))
        cv.waitKey(1)
