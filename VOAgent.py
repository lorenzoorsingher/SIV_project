import pdb
import numpy as np
import math
import cv2 as cv

from common import *
from position import Position

import warnings

warnings.filterwarnings("error")


class VOAgent:
    """
    This class contains the functionalities to run visual odometry.
    Feature matching, essential matrix computation and decomposition
    """

    def __init__(
        self,
        mtx,
        dist,
        buf_size=1,
        matcher_method=SIFT_BF_LOWE,
        num_feat=500,
        scale_factor=1,
        denoise=0,
        debug=True,
    ):
        """
        Initialize the Odometry object

        Parameters
        ----------
        mtx (ndarray): Camera matrix
        dist (ndarray): Distortion coefficients
        buf_size (int): Frames buffer size
        matcher_method (int): Feature matching method
        scale_factor (float): Scale factor for images
        denoise (int): Amount of blur to apply to the images (should be positive and odd)
        debug (bool): Flag to enable image debugging
        """
        self.prevFrame = None
        self.prevDes = None
        self.prevKp = None
        self.position = Position()

        tmp_mtx = mtx * scale_factor
        tmp_mtx[-1][-1] = 1.0
        self.mtx = tmp_mtx
        self.dist = dist
        self.proj = np.hstack([self.mtx, np.array([[0], [0], [0]])])

        self.scale_factor = scale_factor
        self.matcher_method = matcher_method
        self.num_feat = num_feat
        self.buf_size = buf_size
        self.denoise = denoise

        self.debug = debug

    def next_frame(self, nextFrame, abs_scale=1) -> np.ndarray:
        """
        This function processes the last frame and updates the agent position
        running visual odometry between two consecutive frames.

        Parameters
        ----------
        nextFrame (ndarray): frame to be processed
        abs_scale (float): absolute scale if provided from ground truth

        Returns
        -------
        ndarray: The updated agent world position
        """

        # fill frames buffer

        img2 = cv.cvtColor(nextFrame, cv.COLOR_BGR2GRAY)

        if self.scale_factor != 1:
            img2 = cv.resize(img2, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

        if self.denoise > 0:
            img2 = cv.GaussianBlur(img2, (self.denoise, self.denoise), 0)

        # undistort images id dist vector is present
        if not self.dist is None:
            uimg2 = cv.undistort(img2, self.mtx, self.dist)
        else:
            uimg2 = img2

        # set matcher method
        if self.matcher_method == SIFT_BF_LOWE:
            matcher = self.SIFT_BF_LOWE
        elif self.matcher_method == SIFT_FLANN_SORT:
            matcher = self.SIFT_FLANN_SORT
        elif self.matcher_method == ORB_BF_SORT:
            matcher = self.ORB_BF_SORT
        elif self.matcher_method == ORB_BF_LOWE:
            matcher = self.ORB_BF_LOWE
        elif self.matcher_method == ORB_FLANN_LOWE:
            matcher = self.ORB_FLANN_LOWE
        elif self.matcher_method == SIFT_FLANN_LOWE:
            matcher = self.SIFT_FLANN_LOWE

        if self.prevDes is None:

            self.prevFrame = uimg2
            pFrame1, pFrame2, self.prevKp, self.prevDes = matcher(
                None,
                None,
                None,
                uimg2,
            )
            return np.eye(4, 4, dtype=np.float64)

        # get matched points
        pFrame1, pFrame2, self.prevKp, self.prevDes = matcher(
            self.prevKp,
            self.prevDes,
            self.prevFrame,
            uimg2,
        )
        # if len(self.prevDes) > 5000:
        #     print(len(self.prevDes))

        # extract rotation and translation
        R, t, bad_data = self.epipolar_computation(pFrame1, pFrame2)

        # update agent position
        pose = self.position.update_pos(R, t, bad_data, abs_scale)

        self.prevFrame = nextFrame
        return pose

    def epipolar_computation(self, pFrame1, pFrame2) -> tuple:
        """
        Computes the essential matrix and extracts rotation and translation
        from the points matched between two consecutive frames.

        Parameters
        ----------
        - pFrame1 (ndarray): points from first frame
        - pFrame2 (ndarray): points from second frame

        Returns
        -------
        - R (ndarray): rotation matrix
        - t (ndarray): translation vector
        - bad_data (bool): flag to indicate if the data is bad
        """
        bad_data = False
        if len(pFrame1) >= 6 and len(pFrame2) >= 6:
            E, _ = cv.findEssentialMat(
                pFrame1,
                pFrame2,
                self.mtx,
                threshold=1,
                method=cv.RANSAC,
            )

            # TODO: pick better solution
            # _, R, t, mask = cv.recoverPose(E, pFrame1, pFrame2)
            # t = t[:, 0]
            R, t = self.decomp_essential_mat(
                E,
                np.array(pFrame1, dtype=np.float32),
                np.array(pFrame2, dtype=np.float32),
                self.mtx,
                self.proj,
            )

            if abs(t.mean()) > 10:
                bad_data = True
        else:
            R = t = None
            bad_data = True

        return R, t, bad_data

    def decomp_essential_mat(self, E, q1, q2, mtx, initP) -> list:
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        mtx (ndarray): The camera intrinsic matrix
        initP (ndarray): Initial projection matrix

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """

        # Decompose the essential matrix
        R1, R2, t = cv.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = self.sum_z_cal_relative_scale(R, t, mtx, q1, q2, initP)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]

        R1, t = right_pair
        t = t * relative_scale
        return [R1, t]

    def sum_z_cal_relative_scale(self, R, t, mtx, q1, q2, initP) -> tuple:
        """
        Computes the number of points in front of the "cameras" (in our case
        it's not two cameras taking two frames but just one camera taking consecutive
        frames from different positions) to determine which pair of R and t coming
        from the essential matrix decomp is the valid one.

        Parameters
        ----------
        R (ndarray): Rotation matrix
        t (ndarray): Translation vector
        mtx (ndarray): The camera intrinsic matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        initP (ndarray): Initial projection matrix

        Returns
        -------
        sum_of_pos_z_Q1 + sum_of_pos_z_Q2 (int): Number of points in front of the cameras
        relative_scale (float): The relative scale
        """
        # Get the transformation matrix
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        # Make the projection matrix
        P = np.matmul(np.concatenate((mtx, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points w.r.t. the first camera
        hom_Q1 = cv.triangulatePoints(initP, P, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

        # TODO verify correctness of relative scale
        # Form point pairs and calculate the relative scale

        try:
            relative_scale = np.nanmean(
                np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
                / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
            )
        except RuntimeWarning:
            relative_scale = 1
            pass

        if math.isnan(relative_scale):
            relative_scale = 1

        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

    # TODO: refine code for feature matching

    def draw_matches(self, img1, kp1, img2, kp2, good):
        """
        Draw matches between two images

        Parameters
        ----------
        img1 (ndarray): First image
        kp1 (ndarray): Keypoints in first image
        img2 (ndarray): Second image
        kp2 (ndarray): Keypoints in second image
        """
        if self.debug:
            img3 = cv.drawMatchesKnn(
                img1,
                kp1,
                img2,
                kp2,
                good,
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv.namedWindow("feature_matching", cv.WINDOW_NORMAL)
            cv.imshow("feature_matching", img3)

    def SIFT_BF_LOWE(self, kp1, des1, prevFrame, nextFrame):
        # Initiate SIFT detector
        sift = cv.SIFT_create(nfeatures=self.num_feat)
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(nextFrame, None)

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append([m])

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        return pFrame1, pFrame2, kp2, des2

    def SIFT_FLANN_LOWE(self, kp1, des1, prevFrame, nextFrame):
        # Initiate SIFT detector
        sift = cv.SIFT_create(nfeatures=self.num_feat)
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(nextFrame, None)

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        # FLANN parameters
        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=5,
        )

        search_params = {}  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # matches = flann.match(des1, des2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance:
                matchesMask[i] = [1, 0]
                good.append([m])

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        return pFrame1, pFrame2, kp2, des2

    def SIFT_FLANN_SORT(self, kp1, des1, prevFrame, nextFrame):
        # Initiate SIFT detector
        sift = cv.SIFT_create(nfeatures=self.num_feat)
        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(nextFrame, None)

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        # FLANN parameters
        index_params = dict(
            algorithm=FLANN_INDEX_KDTREE,
            trees=3,
        )

        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(des1, des2)

        good = []

        matches = sorted(matches, key=lambda x: x.distance)

        good = [[x] for x in matches[:200]]

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        return pFrame1, pFrame2, kp2, des2

    def ORB_BF_LOWE(self, kp1, des1, prevFrame, nextFrame):
        orb = cv.ORB_create(nfeatures=self.num_feat)
        # find the keypoints and descriptors with ORB
        kp2, des2 = orb.detectAndCompute(nextFrame, None)

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        # Match descriptors.
        # returns a list of k-ouples of DMatch objects
        # each DMatch object contains the indices of the descriptors in the two images that match and the distance between them
        matches = bf.knnMatch(des1, des2, k=2)

        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        return pFrame1, pFrame2, kp2, des2

    def ORB_BF_SORT(self, kp1, des1, prevFrame, nextFrame):
        orb = cv.ORB_create(nfeatures=self.num_feat, scoreType=cv.ORB_FAST_SCORE)
        # find the keypoints and descriptors with ORB
        kp2, des2 = orb.detectAndCompute(nextFrame, None)

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        # returns a list of k-ouples of DMatch objects
        # each DMatch object contains the indices of the descriptors in the two images that match and the distance between them
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test
        good = [[x] for x in matches[:200]]

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        # breakpoint()
        return pFrame1, pFrame2, kp2, des2

    def ORB_FLANN_LOWE(self, kp1, des1, prevFrame, nextFrame):
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=self.num_feat, scoreType=cv.ORB_FAST_SCORE)
        # find the keypoints and descriptors with ORB
        kp2, des2 = orb.detectAndCompute(nextFrame, None)
        # Use FLANN to Find the Best Keypoint Matches

        if kp1 is None or des1 is None:
            return [], [], kp2, des2

        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # was 12
            key_size=12,  # was 20
            multi_probe_level=1,
        )  # was 2

        search_params = dict(checks=32)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # Added this per the Q & A
        if des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2:
            # breakpoint()
            # des1 = np.float32(des1)
            # des2 = np.float32(des2)
            matches = flann.knnMatch(des1, des2, k=2)

        # Store the Keypoint Matches that Pass Lowe's Ratio Test
        good = []

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append([m])

            except ValueError:
                return [], [], None, None

        self.draw_matches(prevFrame, kp1, nextFrame, kp2, good)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)
        return pFrame1, pFrame2, kp2, des2
