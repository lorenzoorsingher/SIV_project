import cv2 as cv
import numpy as np
import pdb

chessboard_size = [7,5,30,22]
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
board = cv.aruco.CharucoBoard(
            (chessboard_size[0],
            chessboard_size[1]),
            chessboard_size[2],
            chessboard_size[3],
            aruco_dict,
        )

def get_corners_charuco(image):

        """
        Charuco corners detection.
        Run a marker detection for the charuco board, interpolates them
        to get the chessboard corners and return two parallel arrays
        containing the corners coordinates and corresponding corners IDs
        """

        print("[corner detection] starting...")

        # sub pixel corner detection criterion
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict)
        imsize = None
        if len(corners) > 0:
            # sub pixel refinement
            for corner in corners:
                cv.cornerSubPix(
                    gray,
                    corner,
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )
            # interpolate charuco markers detections
            res2 = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)

            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                corners = res2[1]
                ids = res2[2]
                imsize = gray.shape

        print("[corner detection] finished.")

        return corners, ids, imsize


video_path = "data/VID_20231205_132133.mp4"
cap = cv.VideoCapture(video_path)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

while True:
     
     ret, frame = cap.read()

     corners, ids, imsize = get_corners_charuco(frame)

     frame = cv.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0,0,255))

     cv.imshow("frame", frame)
     cv.waitKey(1)