import cv2 as cv
import numpy as np
import json
import pdb
import os


def get_corners(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboard_size,
        cv.CALIB_CB_ADAPTIVE_THRESH
        + cv.CALIB_CB_FAST_CHECK
        + cv.CALIB_CB_NORMALIZE_IMAGE,
    )
    print(ret)
    corners2 = None
    if ret == True:
        # objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(image, chessboard_size, corners2, ret)

    cv.imshow("img", image)
    cv.waitKey(1)
    # print("[corner detection] finished.")

    return ret, corners2


chessboard_size = [7, 5]
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)


imgs_path = "data/calib_wide_checkerboard1/images"
calib_path = "camera_data/calib_tum.json"
cv.namedWindow("frame", cv.WINDOW_NORMAL)

allCorners = []
objPoints = []
imgs_names = os.listdir(imgs_path)
# imgs_names.sort()
for name in imgs_names[:100]:
    im_path = imgs_path + "/" + name
    frame = cv.imread(im_path)
    ret, corners = get_corners(frame)
    if ret:
        objPoints.append(objp)
        allCorners.append(corners)


# allCorners = np.array(allCorners,dtype=np.float32)

num_img = 50
corners = []
idxes = np.random.choice(len(allCorners), min(num_img, len(allCorners)), replace=False)
for inx in idxes:
    corners.append(allCorners[inx])

ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv.calibrateCamera(
    objPoints, allCorners, frame.shape[:2:][::-1], None, None
)


# cv.namedWindow("und", cv.WINDOW_NORMAL)
# for name in imgs_names:
#     im_path = imgs_path + "/" + name
#     frame = cv.imread(im_path)

#     undistorted = cv.undistort(frame, camera_matrix, distortion_coefficients)
#     cv.imshow("und", np.hstack([frame, undistorted]))
#     cv.waitKey(0)
# breakpoint()

with open(calib_path, "w", encoding="utf-8") as f:
    json.dump(
        [camera_matrix.tolist(), distortion_coefficients.tolist()],
        f,
        ensure_ascii=False,
        indent=4,
    )

breakpoint()
