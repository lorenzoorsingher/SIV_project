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

calib_path = "camera_data/calib_tum.json"
imgs_path = "data/sequence_14/images"
video_path = "data/room_tour.MOV"

data = json.load(open(calib_path))
mtx = np.array(data[0])
dist = np.array(data[1])
proj = np.hstack([mtx, np.array([[0], [0], [0]])])


cv.namedWindow("map", cv.WINDOW_NORMAL)
cv.namedWindow("gt_map", cv.WINDOW_NORMAL)

np.set_printoptions(formatter={"all": lambda x: str(x)})


odo = Odometry(mtx, dist, 2)

track_map = np.zeros((600, 600, 3))


MODE = "kitti"
if MODE == "imgs":
    im_paths = [imgs_path + "/" + name for name in os.listdir(imgs_path)]
    im_paths.sort()
    idx = 0
if MODE == "video":
    cap = cv.VideoCapture(video_path)
if MODE == "kitti":
    do_images = "data/data_odometry_gray/dataset/sequences"
    do_poses = "data/data_odometry_poses/dataset/poses"

    kl = kittiLoader(do_images, do_poses, 0)
    maxdist = int(kl.get_maxdist())


gt_map = np.zeros((maxdist * 2 + 10, maxdist * 2 + 10, 3))


def update_gtmap(pose, gt_map):
    x, y, z = pose.T[-1]
    R = pose.T[:3].T
    rot = odo.position.rotationMatrixToEulerAngles(R) * (180 / np.pi)

    #    breakpoint()
    gt_map = cv.circle(
        gt_map,
        (int(x) + maxdist, int(z) + maxdist),
        1,
        (255, 255, 0),
        2,
    )

    # y_r = -np.cos(odo.position.cumul_R[1]) * 100
    # x_r = -np.sin(odo.position.cumul_R[1]) * 100

    # # print(np.sin(odo.position.cumul_R[1]))

    # # track_map = cv.circle(track_map, (int(x_r)+300, int(y_r)+300), 1,(255,255,0),2)
    # # track_map = np.zeros((600,600))
    # track_map = cv.circle(
    #     track_map,
    #     (int(odo.position.x) + 300, int(odo.position.y) + 300),
    #     1,
    #     (255, 255, 0),
    #     2,
    # )
    # track_map_tmp = copy(track_map)
    # track_map_tmp = cv.line(
    #     track_map_tmp,
    #     (
    #         int(x_r) + int(odo.position.x) + 300,
    #         int(y_r) + int(odo.position.y) + 300,
    #     ),
    #     (int(odo.position.x) + 300, int(odo.position.y) + 300),
    #     (0, 0, 255),
    #     2,
    # )

    # track_map_tmp = cv.line(
    #     track_map_tmp,
    #     (
    #         int(x_r) + 300,
    #         int(y_r) + 500,
    #     ),
    #     (300, 500),
    #     (0, 255, 0),
    #     2,
    # )
    print("---")
    print(x, "\t", y, "\t", z)
    print(rot)


def update_map(pose, track_map):
    pass


while True:
    if MODE == "video":
        ret, frame = cap.read()
    if MODE == "imgs":
        frame = cv.imread(im_paths[idx])
        idx += 1
        ret = True
    if MODE == "kitti":
        frame, pose = kl.next_frame()
        update_gtmap(pose, gt_map)
        ret = True

    if not ret:
        break
    odo.next_frame(frame)

    y_r = -np.cos(odo.position.cumul_R[1]) * 100
    x_r = -np.sin(odo.position.cumul_R[1]) * 100

    # print(np.sin(odo.position.cumul_R[1]))

    # track_map = cv.circle(track_map, (int(x_r)+300, int(y_r)+300), 1,(255,255,0),2)
    # track_map = np.zeros((600,600))
    track_map = cv.circle(
        track_map,
        (int(odo.position.x) + 300, int(odo.position.y) + 300),
        1,
        (255, 255, 0),
        2,
    )
    track_map_tmp = copy(track_map)
    track_map_tmp = cv.line(
        track_map_tmp,
        (
            int(x_r) + int(odo.position.x) + 300,
            int(y_r) + int(odo.position.y) + 300,
        ),
        (int(odo.position.x) + 300, int(odo.position.y) + 300),
        (0, 0, 255),
        2,
    )

    track_map_tmp = cv.line(
        track_map_tmp,
        (
            int(x_r) + 300,
            int(y_r) + 500,
        ),
        (300, 500),
        (0, 255, 0),
        2,
    )
    # print(cumul_R)
    # breakpoint()
    if True:
        # cv.imshow("frame", np.hstack([uimg1,uimg2]))
        cv.imshow("map", track_map_tmp)
        cv.imshow("gt_map", gt_map)

        cv.waitKey(1)
# prevFrame = newFrame
