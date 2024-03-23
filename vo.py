import os
import json
from copy import copy
import time

import cv2 as cv
import numpy as np
from tqdm import tqdm

from common import *
from VOAgent import VOAgent
from kitti_loader import KittiLoader
from setup import get_args
from evaluation import compute_error, save_metrics, save_metrics_video

np.set_printoptions(formatter={"all": lambda x: str(x)})
# get arguments
args = get_args()

DEBUG = not args["no_debug"]
FRAMESKIP = args["frameskip"]
MODE = args["mode"]
SEQUENCE = args["sequence"]
STEPS = args["steps"]
FEAT_MATCHER = args["feat_match"]
SCALE = args["scale_factor"]
DENOISE = args["denoise"]
NFEAT = args["num_feat"]


out_path = args["output"]
calib_path = args["calib_path"]
video_path = args["video_path"]
do_images = args["kitti_imgs"] + "/dataset/sequences"
do_poses = args["kitti_poses"] + "/dataset/poses"
map_size_default = args["map_size"]


# crete output path if not set
if out_path == "":
    out_path = os.getcwd() + "/data/output/run_" + str(time.time())[:-8]
    if not os.path.exists(out_path):
        os.makedirs(out_path)


if DEBUG:
    cv.namedWindow("gt_map", cv.WINDOW_NORMAL)


if MODE == "video":

    cap = cv.VideoCapture(video_path)

    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mtx = np.array(data[0])
    dist = np.array(data[1])
    if STEPS == -1:
        STEPS = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    (max_x, min_x, max_z, min_z) = (
        map_size_default,
        -map_size_default,
        map_size_default,
        -map_size_default,
    )
if MODE == "kitti":

    kl = KittiLoader(do_images, do_poses, SEQUENCE)
    mtx, dist = kl.get_params()
    maxdist = int(kl.get_maxdist())
    kl.set_idx(0)
    if STEPS == -1:
        STEPS = kl.get_seqlen()
    (max_x, min_x, max_z, min_z) = kl.get_extremes()


# create Visual Odometry Agent
odo = VOAgent(
    mtx,
    dist,
    buf_size=1,
    matcher_method=FEAT_MATCHER,
    num_feat=NFEAT,
    scale_factor=SCALE,
    denoise=DENOISE,
    debug=DEBUG,
)

# create and prepare maps
margin = 50
max_x += margin
min_x -= margin
max_z += margin
min_z -= margin

size_z = max_z - min_z
size_x = max_x - min_x
map_size = (size_z, size_x, 3)
origin = (-min_x, -min_z)

track_map = np.full(map_size, 255, dtype=np.uint8)
updated_gt_map = np.full(map_size, 255, dtype=np.uint8)
if MODE == "kitti":
    track_map = draw_gt_map(track_map, origin, kl)

# initialize poses
old_gt_pose = np.eye(3, 4, dtype=np.float64)
old_agent_pose = np.eye(3, 4, dtype=np.float64)

est_poses = []
gt_poses = []
errors = []

abs_scale = 1

start_time = time.time()

for tqdm_idx in tqdm(range(STEPS // FRAMESKIP)):

    if MODE == "video":
        for i in range(FRAMESKIP):
            ret, frame = cap.read()
            if not ret:
                break
        gt_pose = np.eye(3, 4, dtype=np.float64)

    if MODE == "kitti":
        for i in range(FRAMESKIP):
            frame, gt_pose = kl.next_frame()
        # compute absolute scale from ground truth
        abs_scale = np.linalg.norm(old_gt_pose[:, 3] - gt_pose[:, 3])

    # compute agent pose
    agent_pose = odo.next_frame(frame, abs_scale)[:-1]

    est_poses.append(copy(agent_pose))
    gt_poses.append(copy(gt_pose))

    # error evaluation
    err = compute_error(agent_pose, gt_pose, old_agent_pose, old_gt_pose)
    errors.append(err)

    old_gt_pose = gt_pose.copy()
    old_agent_pose = agent_pose.copy()

    if DEBUG:
        color = (0, 200, 0)
        # modulate color according to error
        color = get_color(err, range=(0, 1))

        updated_track_map = update_map(agent_pose, track_map, origin, color)
        cv.imshow("gt_map", updated_track_map)
        cv.imwrite("output/map_" + str(tqdm_idx) + ".png", updated_track_map)
        cv.waitKey(1)

run_time = time.time() - start_time

steps_sec = STEPS / run_time

if MODE == "kitti":
    # only runs if output path is set
    if out_path != "":
        save_metrics(
            est_poses,
            gt_poses,
            errors,
            settings=args,
            output_path=out_path,
            steps_sec=steps_sec,
        )
else:
    save_metrics_video(est_poses, args, out_path, steps_sec)
