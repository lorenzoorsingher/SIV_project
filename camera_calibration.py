import cv2 as cv
import numpy as np
import json
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

        #print("[corner detection] starting...")

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

        #print("[corner detection] finished.")

        return corners, ids, imsize

def calibrate_camera( allCorners, allIds, imsize):
        """
        - allCorners:   list where every element is a list of points corresponding to
                        the corners detected in a single frame during the capturing
        - allIds:   list where every element is a list of the IDs corresponding to
                    the corners in the allCorners list
        - imsize:   image size
        - board:    charuco board object

        Run the camera calibration algorithm using the datapoints extracted previously
        from the charco dump json file. Intrinsic guess is used to try and help
        converge faster to the optimal intrinsic matrix
        """

        print("[intrinsic calibration] starting...")

        cameraMatrixInit = np.array(
            [
                [1000., 0.0, imsize[0] / 2.0],
                [0.0, 1000., imsize[1] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distCoeffsInit = np.zeros((5, 1))
        flags = cv.CALIB_USE_INTRINSIC_GUESS
        #breakpoint()
        (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
            stdDeviationsIntrinsics,
            stdDeviationsExtrinsics,
            perViewErrors,
        ) = cv.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 100, 0.00001),
        )

        print("[intrinsic calibration] finished.")

        return (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
        )


video_path = "data/board_calib_1.mp4"
calib_path = "camera_data/calib.json"
cap = cv.VideoCapture(video_path)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

allCorners = []
allIds = []
while cap.isOpened():
     
    ret, frame = cap.read()
    if ret is False:
         break
    corners, ids, imsize = get_corners_charuco(frame)
    
    if not isinstance(corners, tuple):
        if len(corners) >= 6:
            allCorners.append(corners)
            allIds.append(ids)
            frame = cv.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0,0,255))

    if False:
        cv.imshow("frame", frame)
        cv.waitKey(1)

#allCorners = np.array(allCorners,dtype=np.float32)

num_img = 50
corners = []
ids = []
idxes = np.random.choice(
                len(allCorners), min(num_img, len(allCorners)), replace=False
            )
for inx in idxes:
    corners.append(allCorners[inx])
    ids.append(allIds[inx])
ret, camera_matrix, distortion_coefficients,_,_=calibrate_camera(allCorners=corners,allIds=ids,imsize=imsize)


with open(calib_path, "w", encoding="utf-8") as f:
    json.dump([camera_matrix.tolist(), distortion_coefficients.tolist()], f, ensure_ascii=False, indent=4)

breakpoint()