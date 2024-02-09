import os
import json
from copy import copy

import cv2 as cv
import numpy as np

from common import *
from odometry import VOAgent
from kitti_loader import kittiLoader


imgs_path = "data/sequence_14/images"

calib_path = "camera_data/calib.json"
maxdist = 300

# cv.namedWindow("track_map", cv.WINDOW_NORMAL)
cv.namedWindow("gt_map", cv.WINDOW_NORMAL)
cv.namedWindow("frames", cv.WINDOW_NORMAL)

np.set_printoptions(formatter={"all": lambda x: str(x)})

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


maxdist = int(maxdist * 1.5)

gt_map = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))
track_map2 = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))
updated_gt_map = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))

odo = VOAgent(mtx, dist, buf_size=1, matcher_method=SIFT_FLANN)


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


while True:
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
        ret = True

    if not ret:
        break

    odo.next_frame(frame)

    updated_track_map2 = update_map(odo.position.world_pose[:-1], track_map2)

    if True:
        cv.imshow("gt_map", np.hstack([updated_gt_map, updated_track_map2]))

        cv.waitKey(1)
# prevFrame = newFrame
