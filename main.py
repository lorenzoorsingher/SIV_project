import cv2 as cv
import numpy as np
import json
import pdb

video_path = "data/walk_buc.mp4"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)



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
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    if True:
        #img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.namedWindow("ORB",cv.WINDOW_NORMAL)
        cv.imshow("ORB",img3)

calib_path = "camera_data/calib.json"

data = json.load(open(calib_path))
mtx = np.array(data[0])
dist = np.array(data[1])





ret, prevFrame = cap.read()
while cap.isOpened():

    ret, newFrame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES,0)
        ret, newFrame = cap.read()
    img1 = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)
    
    pFrame1, pFrame2 = ORB_BF(img1,img2)

    E, mask = cv.findEssentialMat(np.array(pFrame1, dtype=np.float32),np.array(pFrame2, dtype=np.float32),mtx,dist,mtx,dist,cv.RANSAC, 0.999, 1.0,)

    breakpoint()
    #cv.imshow("frame", np.hstack([frame1,frame2]))

    cv.waitKey(1)

    prevFrame = newFrame
