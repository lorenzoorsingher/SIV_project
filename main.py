import cv2 as cv
import numpy as np
import json
import pdb
import math
from common import *
from copy import copy
import os
from odometry import Odometry


calib_path = "camera_data/calib_tum.json"
imgs_path = "data/sequence_14/images"
video_path = "data/room_tour.MOV"

data = json.load(open(calib_path))
mtx = np.array(data[0])
dist = np.array(data[1])
proj = np.hstack([mtx, np.array([[0], [0], [0]])])


cv.namedWindow("map", cv.WINDOW_NORMAL)

odo = Odometry(mtx, dist, 2)

track_map = np.zeros((600, 600, 3))


MODE = "imgs"
if MODE == "imgs":
    im_paths = [imgs_path + "/" + name for name in os.listdir(imgs_path)]
    im_paths.sort()
    idx = 0
if MODE == "video":
    cap = cv.VideoCapture(video_path)


while True:
    if MODE == "video":
        ret, frame = cap.read()
    if MODE == "imgs":
        frame = cv.imread(im_paths[idx])
        idx += 1
        ret = True

    if not ret:
        break
    odo.next_frame(frame)

    y_r = -np.cos(odo.position.cumul_R[1]) * 100
    x_r = -np.sin(odo.position.cumul_R[1]) * 100

    print(np.sin(odo.position.cumul_R[1]))

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
        cv.waitKey(1)
# prevFrame = newFrame
