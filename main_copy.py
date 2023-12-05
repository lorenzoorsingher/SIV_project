import cv2 as cv
import numpy as np

video_path = "data/VID_20231205_104636.mp4"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

cv.imshow("frame", np.hstack([frame1,frame2]))

cv.waitKey(0)