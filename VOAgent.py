import numpy as np
import math
import cv2 as cv

from common import *
from position import Position


class VOAgent:
    """
    This class contains the functionalities to run visual odometry.
    Feature matching, essential matrix computation and decomposition
    """

    def __init__(self, mtx, dist, buf_size=1, matcher_method=SIFT_KNN):
        """
        Initialize the Odometry object

        Parameters
        ----------
        mtx (ndarray): Camera matrix
        dist (ndarray): Distortion coefficients
        buf_size (int): Frames buffer size
        matcher_method (int): Feature matching method
        """
        self.frame_buffer = []
        self.position = Position()
        self.mtx = mtx
        self.dist = dist
        self.proj = np.hstack([mtx, np.array([[0], [0], [0]])])
        self.matcher_method = matcher_method
        # breakpoint()
        self.buf_size = buf_size

    def next_frame(self, lastFrame, abs_scale=1) -> np.ndarray:
        """
        This function processes the last frame and updates the agent position
        running visual odometry between two consecutive frames.

        Parameters
        ----------
        lastFrame: frame to be processed
        """

        # fill frames buffer
        if len(self.frame_buffer) < self.buf_size:
            self.frame_buffer.append(lastFrame)
            return np.eye(4, 4, dtype=np.float64)

        # extract and prepare first and last frames
        firstFrame = self.frame_buffer.pop(0)
        self.frame_buffer.append(lastFrame)
        img1 = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY)

        # undistort images id dist vector is present
        if not self.dist is None:
            uimg1 = cv.undistort(img1, self.mtx, self.dist)
            uimg2 = cv.undistort(img2, self.mtx, self.dist)
        else:
            uimg1 = img1
            uimg2 = img2

        # set matcher method
        if self.matcher_method == SIFT_KNN:
            matcher = self.SIFT_KNN
        elif self.matcher_method == SIFT_FLANN:
            matcher = self.SIFT_FLANN
        elif self.matcher_method == ORB_BF:
            matcher = self.ORB_BF
        elif self.matcher_method == ORB_FLANN:
            matcher = self.ORB_FLANN

        # get matched points
        pFrame1, pFrame2 = matcher(uimg1, uimg2)

        # extract rotation and translation
        R, t, bad_data = self.epipolar_computation(pFrame1, pFrame2)

        # update agent position
        pose = self.position.update_pos(R, t, bad_data, abs_scale)

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

            # TODO: pick best solution
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

        relative_scale = np.nanmean(
            np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
            / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
        )
        # print(relative_scale)
        if math.isnan(relative_scale):
            relative_scale = 1

        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

    # TODO: refine code for feature matching

    def SIFT_KNN(self, img1, img2):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)

        good = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append([m])

        if IMG_DEBUG:
            img3 = cv.drawMatchesKnn(
                img1,
                kp1,
                img2,
                kp2,
                good,
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv.namedWindow("SIFT", cv.WINDOW_NORMAL)
            cv.imshow("SIFT", img3)
        # cv.waitKey(0)
        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        # breakpoint()
        return pFrame1, pFrame2

    def SIFT_FLANN(self, img1, img2):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.append([m])

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv.DrawMatchesFlags_DEFAULT,
        )

        if IMG_DEBUG:
            img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

            cv.namedWindow("SIFT", cv.WINDOW_NORMAL)
            cv.imshow("SIFT", img3)
        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)
        return pFrame1, pFrame2

    def ORB_BF(self, img1, img2):
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test
        good = []
        thr = 300
        for m in matches:
            if (
                abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) < thr
                and abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < thr
            ):
                good.append([m])
        if IMG_DEBUG:
            img3 = cv.drawMatchesKnn(
                img1,
                kp1,
                img2,
                kp2,
                good,
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv.namedWindow("ORB", cv.WINDOW_NORMAL)
            cv.imshow("ORB", img3)

        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)

        # breakpoint()
        return pFrame1, pFrame2

    def ORB_FLANN(self, img1, img2):
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        # Use FLANN to Find the Best Keypoint Matches

        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=12,  # was 12
            key_size=16,  # was 20
            multi_probe_level=2,
        )  # was 2

        # index_params = dict(
        #     algorithm=FLANN_INDEX_AUTOTUNED,
        #     target_precision=0.9,
        #     build_weight=0.01,
        #     memory_weight=0,
        #     sample_fraction=0.0,
        # )
        search_params = dict(checks=100)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # Added this per the Q & A
        if des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2:
            matches = flann.knnMatch(des1, des2, k=2)

        # Store the Keypoint Matches that Pass Lowe's Ratio Test
        good = []

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append([m])

            except ValueError:
                return [], []

        if IMG_DEBUG:
            # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img3 = cv.drawMatchesKnn(
                img1,
                kp1,
                img2,
                kp2,
                good,
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv.namedWindow("ORB", cv.WINDOW_NORMAL)
            cv.imshow("ORB", img3)
        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)
        return pFrame1, pFrame2
