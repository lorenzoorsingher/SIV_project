import cv2 as cv
video_path = "data/VID_20231205_104636.mp4"
cap = cv.VideoCapture(video_path)


cv.namedWindow("frame", cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    cv.imshow("frame", frame)
    cv.waitKey(0)