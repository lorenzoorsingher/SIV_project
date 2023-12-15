import os
import cv2 as cv
import numpy as np
import json
import pdb
import math
import torch
from scipy.spatial.transform import Rotation
from common import *
from copy import copy


def build_poses(line):
    ts = torch.tensor(line[:, 1:4])
    qs = torch.tensor(line[:, 4:])
    rs = torch.eye(4).unsqueeze(0).repeat(qs.shape[0], 1, 1)
    rs[:, :3, :3] = torch.tensor(Rotation.from_quat(qs).as_matrix())
    rs[:, :3, 3] = ts * 1
    poses = rs
    return poses.to(torch.float32)


video_path = "data/room_tour.MOV"
cap = cv.VideoCapture(video_path)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

ground_path = "data/sequence_14/groundtruthSync.txt"
with open(ground_path) as f:
    lines = f.readlines()

images_path = "data/sequence_14/images"
im_paths = [images_path + "/" + name for name in os.listdir(images_path)]
im_paths.sort()

for idx, im in enumerate(im_paths):
    frame = cv.imread(im)
    line = lines[idx]
    cv.imshow("frame", frame)
    cv.waitKey(1)
    print(line)
    breakpoint()
    pose = build_poses(line.split(" "))
    breakpoint()

breakpoint()
