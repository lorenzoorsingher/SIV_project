import cv2 as cv
import numpy as np
import json
import pdb
import math
from common import *
from copy import copy
import os
from odometry import Odometry
from kitti_loader import kittiLoader

from scipy.spatial.transform import Rotation


imgs_path = "data/sequence_14/images"

calib_path = "camera_data/calib.json"
maxdist = 300

# cv.namedWindow("track_map", cv.WINDOW_NORMAL)
cv.namedWindow("gt_map", cv.WINDOW_NORMAL)
# cv.namedWindow("updated_track_map2", cv.WINDOW_NORMAL)

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
    do_calib = "data/data_odometry_poses/dataset/sequences"

    kl = kittiLoader(do_images, do_poses, do_calib, 0)
    mtx, dist = kl.get_params()
    maxdist = int(kl.get_maxdist())


maxdist = int(maxdist * 1.5)

gt_map = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))
track_map2 = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))
updated_gt_map = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))

odo = Odometry(mtx, dist, 2)


def update_map(pose, map):
    x, _, z = pose.T[-1]
    R = pose.T[:3].T

    map = cv.circle(
        map,
        (int(x) + maxdist, int(z) + maxdist),
        1,
        (255, 255, 0),
        2,
    )

    updated_map = copy(map)

    forw = np.array([0, 0, 100, 1])

    newt = np.eye(4, 4)
    newt[:3, :3] = R
    newt[2, 3] = 1
    nx, _, nz, _ = newt @ forw

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

    eulered = odo.position.rotationMatrixToEulerAngles(R) * 180 / np.pi
    cv2.putText(
        updated_map,
        str(eulered[1].round(2)),
        (10, updated_map.shape[0] // 10),
        cv.FONT_HERSHEY_PLAIN,
        updated_map.shape[0] // 100,
        (255, 255, 255),
        updated_map.shape[0] // 100,
        cv2.LINE_AA,
    )
    return updated_map


def update_map2(x, z, map):
    map = cv.circle(
        map,
        (int(x) + maxdist, int(z) + maxdist),
        1,
        (255, 255, 0),
        2,
    )

    return map


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

    # updated_track_map2 = update_map2(
    #     odo.position.world_coo[0], odo.position.world_coo[2], track_map2
    # )

    updated_track_map2 = update_map(odo.position.world_pose[:-1], track_map2)
    # tmpPose = np.zeros((3, 4))
    # tmpPose[:3, :3] = odo.position.cumul_R
    # tmpPose[:3, 3] = odo.position.cumul_t
    # updated_track_map = update_map(tmpPose, track_map)

    if True:
        cv.imshow("gt_map", np.hstack([updated_gt_map, updated_track_map2]))
        # cv.imshow("track_map", updated_track_map)
        # cv.imshow("updated_track_map2", updated_track_map2)

        cv.waitKey(1)
# prevFrame = newFrame
