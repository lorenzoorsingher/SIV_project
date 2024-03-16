import os
import subprocess
import numpy as np
import json
import shutil
import sys
import time

feature_matchers = [
    ("ORB_BF", 0),
    ("ORB_FLANN", 1),
    ("SIFT_FLANN", 2),
    ("SIFT_KNN", 3),
    ("ORB_KNN", 4),
    ("SIFT_FLANN_LOWE", 5),
]
scales = [0.25, 0.5, 1]

denoise = [0]

sequences = [0]

steps = 30

out_path = os.getcwd() + "/data/output/run_" + str(time.time())[:-8]
if not os.path.exists(out_path):
    os.makedirs(out_path)

##########

feature_matchers = [
    ("ORB_BF", 0),
]
scales = [1]

denoise = [3, 5, 7]

sequences = [0]

steps = 400
##########

index = 0
for sequence in sequences:
    for scale in scales:
        for denoise_val in denoise:
            for fm, idx in feature_matchers:
                eval_path = out_path + "/eval_" + str(index)
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)
                index += 1
                print("\n\n----------------------------------------------")
                print(
                    "Running sequence",
                    sequence,
                    " with scale: ",
                    scale,
                    " denoising: ",
                    denoise_val,
                    " feature matcher: ",
                    fm,
                )
                os.system(
                    sys.executable
                    + " vo.py -st "
                    + str(steps)
                    + " -s "
                    + str(sequence)
                    + " -o "
                    + eval_path
                    + " -sf "
                    + str(scale)
                    + " -fm "
                    + str(idx)
                    + " -de "
                    + str(denoise_val)
                    + " -nd"
                )
