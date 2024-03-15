import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from common import *

path = "/home/lollo/Documents/python/siv/SIV_project/data/output/"
dirs = [path + d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
dirs.sort()
latest = dirs[-1]
latest += "/"
dirs = [
    latest + d for d in os.listdir(latest) if os.path.isdir(os.path.join(latest, d))
]

# fig, axs = plt.subplots(3, 3, figsize=(5, 5))
# for ax_idx, ax in enumerate(axs.flatten()):
#     sample_image, sample_label = dataset[ax_idx]

#     ax.axis("off")
#     ax.set_title(f"Class: {sample_label}")
#     ax.imshow(np.asarray(sample_image))

results = {}
for dir in dirs:
    error = json.load(open(dir + "/err.json"))
    settings = json.load(open(dir + "/settings.json"))

    # breakpoint()
    name = (
        FM[int(settings["feat_match"])]
        + "_"
        + str(settings["scale_factor"])
        + "_"
        + str(settings["denoise"])
    )
    results[name] = np.average(error)
breakpoint()
