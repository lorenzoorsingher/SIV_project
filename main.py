import cv2 as cv
import numpy as np
import pdb

video_path = "data/VID_20231205_104636.mp4"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)



ret, prevFrame = cap.read()
while cap.isOpened():

    ret, newFrame = cap.read()
    if not ret:
        cap.set(cv.CAP_PROP_POS_FRAMES,LRtmp-1)
    img1 = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(newFrame, cv.COLOR_BGR2GRAY)
    # Initiate ORB detector
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

    pFrame1 = [kp1[g[0].queryIdx] for g in good]
    pFrame2 = [kp2[g[0].trainIdx] for g in good]



    #cv.imshow("frame", np.hstack([frame1,frame2]))

    cv.waitKey(1)

    prevFrame = newFrame
