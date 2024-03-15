import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from common import *


def draw_maps(all_poses):
    colors = [
        (255, 0, 100),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    max_x = 0
    min_x = np.inf
    max_z = 0
    min_z = np.inf

    for poses in all_poses:

        max_x = max(max_x, int(np.array(poses)[:, 3].max()))
        min_x = min(min_x, int(np.array(poses)[:, 3].min()))
        max_z = max(max_z, int(np.array(poses)[:, 11].max()))
        min_z = min(min_x, int(np.array(poses)[:, 11].min()))
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
    map = np.full(map_size, 255, dtype=np.uint8)

    for idx, poses in enumerate(all_poses):
        for pose in poses:
            x = pose[3]
            z = pose[11]
            # update trace of map
            map = cv.circle(
                map,
                (int(x) + origin[0], int(z) + origin[1]),
                1,
                colors[idx],
                (size_z * size_x) // 80000,
            )
    return map


path = os.getcwd() + "/data/output/"

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

aggregate = {}

for dir in dirs:
    settings = json.load(open(dir + "/settings.json"))
    name = (
        FM[int(settings["feat_match"])]
        + "_"
        + str(settings["scale_factor"])
        + "_"
        + str(settings["denoise"])
    )

    if name not in aggregate:
        aggregate[name] = {"errors": [], "sequences": []}

    if settings["sequence"] not in aggregate[name]["sequences"]:
        aggregate[name]["sequences"].append(settings["sequence"])

        error = json.load(open(dir + "/err.json"))
        aggregate[name]["errors"] += error


final = []
for key, settings in aggregate.items():

    aggregate[key]["average_error"] = np.average(settings["errors"])
    print(key, " ", aggregate[key]["average_error"].round(3))
    final.append([key, aggregate[key]["average_error"]])

sorted = sorted(final, key=lambda x: x[1])
tops = [x[0] for x in sorted]
seqid = 0

all_poses = []
for dir in dirs:

    settings = json.load(open(dir + "/settings.json"))
    name = (
        FM[int(settings["feat_match"])]
        + "_"
        + str(settings["scale_factor"])
        + "_"
        + str(settings["denoise"])
    )
    if name in tops[:3]:
        if settings["sequence"] == seqid:
            if len(all_poses) == 0:
                all_poses.append(json.load(open(dir + "/gt.json")))
            all_poses.append(json.load(open(dir + "/est.json")))

map = draw_maps(all_poses)

cv.imshow("map", map)
cv.waitKey(0)
breakpoint()
