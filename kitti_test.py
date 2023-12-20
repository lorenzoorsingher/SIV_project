import os
import pdb
import numpy as np
import cv2 as cv
from kitti_loader import kittiLoader


do_images = "data/data_odometry_gray/dataset/sequences"
do_poses = "data/data_odometry_poses/dataset/poses"
do_calib = "data/data_odometry_poses/dataset/sequences"

kl = kittiLoader(do_images, do_poses, do_calib, 0)

cv.namedWindow("frame", cv.WINDOW_NORMAL)


breakpoint()
# while True:
#     img, pose = kl.next_frame()

#     cv.imshow("frame", img)
#     cv.waitKey(1)
