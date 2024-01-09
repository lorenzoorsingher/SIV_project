import numpy as np
import math
import cv2 as cv

from common import *
from copy import copy


class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.world_coo = np.array([0, 0, 0, 1])
        self.world_pose = np.eye(4, 4)
        self.cumul_R = np.eye(3, 3, dtype=np.float64)
        self.cumul_t = np.array([0, 0, 0], dtype=np.float64)
        self.lastgoodpose = np.eye(4, 4)

    def update_pos(self, R, t, bad_data):
        tmpT = np.eye(4, 4)
        tmpT[:3, :3] = R
        tmpT[:3, 3] = t

        # curr_coo = np.array([self.x, self.y, self.z, 1])
        # self.x, self.y, self.z, _ = np.dot(tmpT, curr_coo)
        # self.world_coo = np.dot(tmpT, np.array([0, 0, 0, 1]))

        eulered = self.rotationMatrixToEulerAngles(R) * 180 / np.pi

        sizestr = "#"
        for i in range(int(abs(eulered[1])) // 1):
            sizestr += "#"

        if abs(eulered[1]) >= 10 or bad_data:
            R = self.lastgoodpose[:3, :3]
            t = self.lastgoodpose[:3, 3]
            sizestr2 = ""
            for i in range(len(sizestr)):
                sizestr2 += "X"
            sizestr = sizestr2
        else:
            self.lastgoodpose = np.eye(4, 4)
            self.lastgoodpose[:3, :3] = R
            self.lastgoodpose[:3, 3] = t

        print(eulered.round(2), "\t", sizestr)

        # Apply transformations
        self.cumul_R = np.dot(R, self.cumul_R)
        self.cumul_t = t + np.dot(R, self.cumul_t)

        # v = R^T * v' - R^T * t

        self.world_pose[:3, :3] = self.cumul_R.T
        self.world_pose[:3, 3] = np.dot(-(self.cumul_R).T, self.cumul_t)

    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


class Odometry:
    def __init__(self, mtx, dist, buf_size=2):
        self.frame_buffer = []
        self.position = Position()
        self.mtx = mtx
        self.dist = dist
        self.proj = np.hstack([mtx, np.array([[0], [0], [0]])])
        # breakpoint()
        self.buf_size = buf_size

    def next_frame(self, lastFrame):
        if len(self.frame_buffer) < self.buf_size:
            self.frame_buffer.append(lastFrame)
            return

        firstFrame = self.frame_buffer.pop(0)
        self.frame_buffer.append(lastFrame)
        img1 = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY)

        if not self.dist is None:
            uimg1 = cv.undistort(img1, self.mtx, self.dist)
            uimg2 = cv.undistort(img2, self.mtx, self.dist)
        else:
            uimg1 = img1
            uimg2 = img2

        pFrame1, pFrame2 = self.SIFT_KNN(uimg1, uimg2)

        R, t, bad_data = self.epipolarComputation(pFrame1, pFrame2)

        self.position.update_pos(R, t, bad_data)

        cv.imshow("frames", np.vstack([img2, uimg1]))

    # def homographyComputation(self, pFrame1, pFrame2):
    #     M, mask = cv.findHomography(pFrame1, pFrame2, cv.RANSAC, 5.0)
    #     return M

    def epipolarComputation(self, pFrame1, pFrame2):
        bad_data = False
        if len(pFrame1) >= 6 and len(pFrame2) >= 6:
            E, _ = cv2.findEssentialMat(
                pFrame1,
                pFrame2,
                self.mtx,
                threshold=1,
                method=cv.RANSAC,
            )

            R, t = decomp_essential_mat(
                E,
                np.array(pFrame1, dtype=np.float32),
                np.array(pFrame2, dtype=np.float32),
                self.mtx,
                self.proj,
            )

            if abs(t.mean()) > 10:
                bad_data = True
        else:
            bad_data = True

        return R, t, bad_data

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
        if True:
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
        if True:
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
        FLANN_INDEX_LSH = 6

        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # was 12
            key_size=12,  # was 20
            multi_probe_level=1,
        )  # was 2
        search_params = {}

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

        if True:
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
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        cv.namedWindow("SIFT", cv.WINDOW_NORMAL)
        cv.imshow("SIFT", img3)
        pFrame1 = np.array([kp1[g[0].queryIdx].pt for g in good], dtype=np.float32)
        pFrame2 = np.array([kp2[g[0].trainIdx].pt for g in good], dtype=np.float32)
        return pFrame1, pFrame2
