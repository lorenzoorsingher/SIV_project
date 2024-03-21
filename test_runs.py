import os
import subprocess
import numpy as np
import json
import shutil
import sys
import time

from common import *


out_path = os.getcwd() + "/data/output/run_" + str(time.time())[:-8]
if not os.path.exists(out_path):
    os.makedirs(out_path)

########## SETTINGS SETUP

feature_matchers = [
    # SIFT_FLANN_LOWE,
    SIFT_KNN,
    # ORB_FLANN,
    ORB_KNN,
]
scales = [1]

denoise = [0]

sequences = [0, 8]

nfeatures = {
    SIFT_FLANN_LOWE: [50, 100, 500, 2000, 3000, 6000],
    SIFT_KNN: [50, 100, 500, 2000, 3000, 6000],
    ORB_FLANN: [1000, 2000, 3000, 6000, 8000, 10000],
    ORB_KNN: [1000, 2000, 3000, 6000, 8000, 10000],
}

steps = 300

##########


total_steps = (
    len(sequences) * len(scales) * len(denoise) * len(nfeatures) * len(feature_matchers)
)

index = 0
for sequence in sequences:
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
                        "Running sequence",
                        sequence,
                        " with scale: ",
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
                        + " vo.py -st "
                        + str(steps)
                        + " -s "
                        + str(sequence)
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

os.system(sys.executable + " compare_results.py")
