import os
import subprocess
import numpy as np
import json
import shutil
import sys

feature_matchers = [
    ("ORB_BF", 0),
    ("ORB_FLANN", 1),
    ("SIFT_FLANN", 2),
    ("SIFT_KNN", 3),
    ("ORB_KNN", 4),
    ("SIFT_FLANN_LOWE", 5),
]
scales = [0.25, 0.5, 1]

sequences = [0]

steps = 30

for sequence in sequences:
    for scale in scales:
        for fm, idx in feature_matchers:
            print("\n\n----------------------------------------------")
            print(
                "Running sequence",
                0,
                " with scale: ",
                scale,
                " and feature matcher: ",
                fm,
            )
            os.system(
                sys.executable
                + " vo.py -st "
                + str(steps)
                + " -s "
                + str(sequence)
                + " -sf "
                + str(scale)
                + " -fm "
                + str(idx)
                + " -nd"
            )