import os
import cv2 as cv
import numpy as np


class kittiLoader:
    def __init__(self, do_images, do_poses, sequence_id=0):
        self.do_images = do_images
        self.do_poses = do_poses
        self.sequence_id = sequence_id
        self.cur_idx = 0

        if sequence_id < 10:
            sequence_id = "0" + str(sequence_id)
        else:
            sequence_id = str(sequence_id)

        self.im_paths = do_images + "/" + sequence_id + "/image_0"
        pspath = do_poses + "/" + sequence_id + ".txt"

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

        # breakpoint()

    def get_seqlen(self):
        return len(self.im_paths)

    def get_maxdist(self):
        return np.array(self.poses).max()

    def get_frame(self, idx):
        img = cv.imread(self.im_paths[idx])
        pose = self.poses[idx]
        return img, pose

    def next_frame(self):
        img = cv.imread(self.im_paths[self.cur_idx])
        pose = self.poses[self.cur_idx]
        self.cur_idx = (self.cur_idx + 1) % self.get_seqlen()
        return img, pose

    def set_idx(self, idx):
        self.cur_idx = idx
