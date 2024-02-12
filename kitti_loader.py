import os
import cv2 as cv
import numpy as np


class KittiLoader:
    def __init__(self, do_images: str, do_poses: str, sequence_n: int = 0):
        """
        Load KITTI dataset

        Parameters
        ----------
        do_images (str): Path to the images root directory
        do_poses (str): Path to the poses root directory
        sequence_n (int): Sequence number to be loaded
        """
        self.do_images = do_images
        self.do_poses = do_poses
        self.sequence_id = ("0" + str(sequence_n))[-2:]

        self.cur_idx = 0

        # path to images of sequence
        self.im_paths = do_images + "/" + self.sequence_id + "/image_0"
        # path to poses of sequence
        ps_path = do_poses + "/" + self.sequence_id + ".txt"

        # all single images paths in sequence
        self.im_paths = [
            self.im_paths + "/" + name for name in os.listdir(self.im_paths)
        ]
        self.im_paths.sort()

        # all poses in sequence corresponding to images
        self.poses_tmp = []
        with open(ps_path) as f:
            self.poses_tmp = f.readlines()
            self.poses_tmp = [p.replace("\n", "") for p in self.poses_tmp]

        self.poses = [
            np.array([float(val) for val in p.split(" ")]).reshape(3, 4).round(2)
            for p in self.poses_tmp
        ]

        # load calibration matrix (images are already rectified, so no distortion coefficients are needed)
        self.do_calib = do_images + "/" + self.sequence_id + "/calib.txt"
        calibstr = []

        with open(self.do_calib) as f:
            calibstr = f.readlines()
            calib = [float(num) for num in calibstr[0][4:-1].split(" ")]
            calib = np.array(calib).reshape(3, 4).T[:-1].T
            self.mtx = calib
            self.dist = None
            # breakpoint()

    def get_seqlen(self) -> int:
        """
        Returns the length of the sequence
        """
        return len(self.im_paths)

    def get_maxdist(self) -> int:
        """
        Returns the maximum distance run from the origin by the agent
        """
        return np.array(self.poses).max()

    def get_frame(self, idx) -> tuple:
        """
        Returns the image and pose at the given index
        """
        img = cv.imread(self.im_paths[idx])
        pose = self.poses[idx]
        return img, pose

    def get_params(self) -> tuple:
        """
        Returns the camera matrix and distortion coefficients
        """
        return self.mtx, self.dist

    def next_frame(self) -> tuple:
        """
        Returns the next image and pose in the sequence
        """
        img = cv.imread(self.im_paths[self.cur_idx])
        pose = self.poses[self.cur_idx]
        self.cur_idx = (self.cur_idx + 1) % self.get_seqlen()
        return img, pose

    def set_idx(self, idx):
        self.cur_idx = idx
