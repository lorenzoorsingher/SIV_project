import cv2 as cv
import numpy as np
import json
import pdb

chessboard_size = [7, 5]


def get_corners(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboard_size,
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
        + cv.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ret == True:
        # objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, chessboard_size, corners2, ret)

    cv.imshow("img", img)
    cv.waitKey(1)
    # print("[corner detection] finished.")

    return corners


def calibrate_camera(allCorners, allIds, imsize):
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
            [1000.0, 0.0, imsize[0] / 2.0],
            [0.0, 1000.0, imsize[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    distCoeffsInit = np.zeros((5, 1))
    flags = cv.CALIB_USE_INTRINSIC_GUESS
    # breakpoint()
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


video_path = "data/room_calib.MOV"
calib_path = "camera_data/calib.json"
cap = cv.VideoCapture(video_path)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

allCorners = []
allIds = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        break
    corners = get_corners(frame)
    breakpoint()
    if not isinstance(corners, tuple):
        if len(corners) >= 6:
            allCorners.append(corners)
            allIds.append(ids)
            frame = cv.aruco.drawDetectedCornersCharuco(
                frame, corners, ids, (0, 0, 255)
            )

    if False:
        cv.imshow("frame", frame)
        cv.waitKey(0)

# allCorners = np.array(allCorners,dtype=np.float32)

num_img = 50
corners = []
ids = []
idxes = np.random.choice(len(allCorners), min(num_img, len(allCorners)), replace=False)
for inx in idxes:
    corners.append(allCorners[inx])
    ids.append(allIds[inx])
ret, camera_matrix, distortion_coefficients, R, t = calibrate_camera(
    allCorners=corners, allIds=ids, imsize=imsize
)

with open(calib_path, "w", encoding="utf-8") as f:
    json.dump(
        [camera_matrix.tolist(), distortion_coefficients.tolist()],
        f,
        ensure_ascii=False,
        indent=4,
    )

breakpoint()
