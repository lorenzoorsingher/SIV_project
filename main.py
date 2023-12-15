import cv2 as cv
import numpy as np
import json
import pdb
import math
from common import *
from copy import copy

from odometry import Odometry

video_path = "data/room_tour.MOV"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)


calib_path = "camera_data/calib.json"

data = json.load(open(calib_path))
mtx = np.array(data[0])
dist = np.array(data[1])
proj = np.hstack([mtx, np.array([[0], [0], [0]])])


cv.namedWindow("map", cv.WINDOW_NORMAL)

odo = Odometry(mtx, dist, 4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    odo.next_frame(frame)
    # prevFrame = newFrame
