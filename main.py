import cv2 as cv
import numpy as np
import json
import pdb
import math
from common import *
from copy import copy
video_path = "data/room_tour.MOV"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)

def rotationMatrixToEulerAngles(R) :
 
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def ORB_BF(img1, img2):
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Apply ratio test
    good = []
    thr = 50
    for m in matches:
        if abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) < thr and abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < thr:
            good.append([m])
    if True:
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.namedWindow("ORB",cv.WINDOW_NORMAL)
        cv.imshow("ORB",img3)

    pFrame1 = [kp1[g[0].queryIdx].pt for g in good]
    pFrame2 = [kp2[g[0].trainIdx].pt for g in good]
    return pFrame1, pFrame2

def ORB_FLANN(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # Use FLANN to Find the Best Keypoint Matches
    FLANN_INDEX_LSH = 6

    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,         # was 12
                    key_size = 12,            # was 20
                    multi_probe_level = 1)    # was 2
    search_params = {}

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Added this per the Q & A
    if(des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2):
        matches = flann.knnMatch(des1, des2, k = 2)

    # Store the Keypoint Matches that Pass Lowe's Ratio Test
    good = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.7*n.distance:
                good.append([m])

        except ValueError:
            return [], []

    if True:
        #img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.namedWindow("ORB",cv.WINDOW_NORMAL)
        cv.imshow("ORB",img3)
    pFrame1 = [kp1[g[0].queryIdx].pt for g in good]
    pFrame2 = [kp2[g[0].trainIdx].pt for g in good]
    return pFrame1, pFrame2
calib_path = "camera_data/calib.json"

data = json.load(open(calib_path))
mtx = np.array(data[0])
dist = np.array(data[1])
proj = np.hstack([mtx, np.array([[0],[0],[0]])])


cumul_t = np.array([0,0,0], dtype=np.float64)
cumul_R = np.array([0,0,0], dtype=np.float64)

ret, prevFrame = cap.read()

track_map = np.zeros((600,600,3))
cv.namedWindow("map", cv.WINDOW_NORMAL)


frame_buffer = []

for i in range(4):
    ret, frm = cap.read()
    frame_buffer.append(frm)


cam_x = 0
cam_y = 0

while cap.isOpened():

    # ret, newFrame = cap.read()
    # if not ret:
    #     cap.set(cv.CAP_PROP_POS_FRAMES,0)
    #     ret, newFrame = cap.read()


    firstFrame = frame_buffer.pop(0)
    ret, lastFrame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES,0)
        ret, lastFrame = cap.read()

    frame_buffer.append(lastFrame)

    img1 = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(lastFrame, cv.COLOR_BGR2GRAY)
    
    uimg1 = cv.undistort(img1, mtx, dist)
    uimg2 = cv.undistort(img2, mtx, dist)

    pFrame1, pFrame2 = ORB_FLANN(uimg1,uimg2)
    #print(pFrame1)
    if len(pFrame1)>=6 and len(pFrame2) >= 6:
        E, mask = cv.findEssentialMat(np.array(pFrame1, dtype=np.float32),np.array(pFrame2, dtype=np.float32),mtx,dist,mtx,dist,cv.RANSAC, 0.999, 1.0,)

        R, t = decomp_essential_mat(E, np.array(pFrame1, dtype=np.float32),np.array(pFrame2, dtype=np.float32), mtx, proj)
        #breakpoint()
        if abs(t.mean()) < 10:
            cumul_t += t
            #breakpoint()
            cumul_R += rotationMatrixToEulerAngles(R) 
            #breakpoint()
            y_r = -np.cos(cumul_R[1]) * 100
            x_r = -np.sin(cumul_R[1]) * 100

            cam_y += np.cos(cumul_R[1]) * t[0] 
            cam_x += np.sin(cumul_R[1]) * t[2] 

            print(np.sin(cumul_R[1]))

            # track_map = cv.circle(track_map, (int(x_r)+300, int(y_r)+300), 1,(255,255,0),2)
            #track_map = np.zeros((600,600))
            track_map = cv.circle(track_map, (int(cam_x) + 300, int(cam_y)+300), 1,(255,255,0),2)
            track_map_tmp = copy(track_map)
            track_map_tmp = cv.line(track_map_tmp, (int(x_r)+int(cam_x)+300, int(y_r)+int(cam_y)+300), (int(cam_x)+300,int(cam_y)+300),(0,0,255),2)
            #print(cumul_R)
        #breakpoint()
        if True:
            #cv.imshow("frame", np.hstack([uimg1,uimg2]))
            cv.imshow("map", track_map_tmp)
            cv.waitKey(1)

    #prevFrame = newFrame
