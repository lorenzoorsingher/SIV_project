import os
import pdb
import cv2 as cv
from kitti_loader import kittiLoader


do_images = "data/data_odometry_gray/dataset/sequences"
do_poses = "data/data_odometry_poses/dataset/poses"

kl = kittiLoader(do_images, do_poses, 0)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

while True:
    img, pose = kl.next_frame()

    cv.imshow("frame", img)
    cv.waitKey(1)
