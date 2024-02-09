import os
import cv2 as cv
import numpy as np


class kittiLoader:
    def __init__(self, do_images, do_poses, sequence_n=0):
        self.do_images = do_images
        self.do_poses = do_poses
        self.sequence_id = sequence_n
        self.cur_idx = 0

        if self.sequence_id < 10:
            self.sequence_id = "0" + str(self.sequence_id)
        else:
            self.sequence_id = str(self.sequence_id)

        self.im_paths = do_images + "/" + self.sequence_id + "/image_0"
        pspath = do_poses + "/" + self.sequence_id + ".txt"

        self.im_paths = [
            self.im_paths + "/" + name for name in os.listdir(self.im_paths)
        ]
        self.im_paths.sort()

        self.poses_tmp = []
        with open(pspath) as f:
            self.poses_tmp = f.readlines()
            self.poses_tmp = [p.replace("\n", "") for p in self.poses_tmp]

        self.poses = [
            np.array([float(val) for val in p.split(" ")]).reshape(3, 4).round(2)
            for p in self.poses_tmp
        ]

        self.do_calib = do_images + "/" + self.sequence_id + "/calib.txt"
        calibstr = []

        with open(self.do_calib) as f:
            calibstr = f.readlines()
            calib = [float(num) for num in calibstr[0][4:-1].split(" ")]
            calib = np.array(calib).reshape(3, 4).T[:-1].T
            self.mtx = calib
            self.dist = None
            # breakpoint()

    def get_seqlen(self):
        return len(self.im_paths)

    def get_maxdist(self):
        return np.array(self.poses).max()

    def get_frame(self, idx):
        img = cv.imread(self.im_paths[idx])
        pose = self.poses[idx]
        return img, pose

    def get_params(self):
        return self.mtx, self.dist

    def next_frame(self):
        img = cv.imread(self.im_paths[self.cur_idx])
        pose = self.poses[self.cur_idx]
        self.cur_idx = (self.cur_idx + 1) % self.get_seqlen()
        return img, pose

    def set_idx(self, idx):
        self.cur_idx = idx
