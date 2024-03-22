import os
import subprocess
import numpy as np
import json
import shutil
import sys
import time

from common import *

# TODO EVERYTHING
out_path = os.getcwd() + "/data/output/run_" + str(time.time())[:-8]
if not os.path.exists(out_path):
    os.makedirs(out_path)

########## SETTINGS SETUP

feature_matchers = [
    ORB_FLANN_LOWE,
]
scales = [0.25]

denoise = [0]

video_path = "data/povo/povo_run.mp4"
calib_path = "camera_data/povo_calib.json"

nfeatures = {
    SIFT_FLANN_LOWE: [50, 100, 500, 2000, 3000, 6000],
    SIFT_BF_LOWE: [50, 100, 500, 2000, 3000, 6000],
    ORB_FLANN_LOWE: [2000, 3000, 6000],
    ORB_BF_LOWE: [1000, 2000, 3000, 6000, 8000, 10000],
}


steps = 10

##########
tot_fm = sum([len(x[1]) for x in nfeatures.items() if x[0] in feature_matchers])
total_steps = len(scales) * len(denoise) * len(feature_matchers) * tot_fm

index = 0
for scale in scales:
    for denoise_val in denoise:
        for fm in feature_matchers:
            for nfeat in nfeatures[fm]:
                eval_path = out_path + "/eval_" + str(index)
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)
                index += 1
                print("\n\n----------------------------------------------")
                print(
                    "Running video with scale: ",
                    scale,
                    " denoising: ",
                    denoise_val,
                    " feature matcher: ",
                    FM[fm],
                    " @ ",
                    nfeat,
                    " ",
                    index,
                    "/",
                    total_steps,
                )
                os.system(
                    sys.executable
                    + " vo.py -m video  -st "
                    + str(steps)
                    + " -vp "
                    + video_path
                    + " -cp "
                    + calib_path
                    + " -o "
                    + eval_path
                    + " -sf "
                    + str(scale)
                    + " -fm "
                    + str(fm)
                    + " -nf "
                    + str(nfeat)
                    + " -de "
                    + str(denoise_val)
                    + " -nd"
                )


path = os.getcwd() + "/data/output/"
dirs = [path + d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
dirs.sort()
fullpath = dirs[-1]
fullpath += "/"
dirs = [
    fullpath + d
    for d in os.listdir(fullpath)
    if os.path.isdir(os.path.join(fullpath, d))
]

all_poses = []
for dir in dirs:
    settings = json.load(open(dir + "/settings.json"))
    name = (
        FM[int(settings["feat_match"])]
        + "_"
        + str(settings["num_feat"])
        + "_"
        + str(settings["scale_factor"])
        + "_"
        + str(settings["denoise"])
    )

    if len(all_poses) == 0:
        all_poses.append(json.load(open(dir + "/est.json")))

breakpoint()
