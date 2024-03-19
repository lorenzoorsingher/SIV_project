import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from common import *
import argparse


parser = argparse.ArgumentParser(
    prog="vo.py",
    description="""Run visual odometry on any video, 
    when no arguments are provided the script will run
    in KITTI mode on sequence 0 with SIFT_KNN.""",
)

parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="path to the run folder",
    default="latest",
    metavar="",
)

args = vars(parser.parse_args())


def draw_maps(all_poses):

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green (Dark)
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
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


print(args["path"])
if args["path"] != "latest":
    fullpath = args["path"] + "/"
else:
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
        + str(settings["num_feat"])
        + "_"
        + str(settings["scale_factor"])
        + "_"
        + str(settings["denoise"])
    )

    if name not in aggregate:
        aggregate[name] = {"errors": [], "sequences": []}
        aggregate[name]["steps_sec"] = []

    if settings["sequence"] not in aggregate[name]["sequences"]:
        aggregate[name]["sequences"].append(settings["sequence"])

        error = json.load(open(dir + "/err.json"))
        aggregate[name]["errors"] += error
        aggregate[name]["steps_sec"] += [settings["steps_sec"]]


final = []
for key, value in aggregate.items():
    aggregate[key]["error_avg"] = np.average(value["errors"])
    aggregate[key]["error_std"] = np.std(value["errors"])
    aggregate[key]["error_max"] = np.max(value["errors"])
    aggregate[key]["steps_sec"] = np.average(value["steps_sec"])

    final.append(
        [
            key,
            key.split("_")[-3],
            key.split("_")[-2],
            key.split("_")[-1],
            aggregate[key]["error_avg"],
            aggregate[key]["error_std"],
            aggregate[key]["error_max"],
            aggregate[key]["steps_sec"],
        ]
    )

sorted = sorted(final, key=lambda x: x[4])
tops = [x[0] for x in sorted]

for s in sorted:
    print(s)
seqid = 0

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

    if name in tops[:3]:
        if settings["sequence"] == seqid:
            if len(all_poses) == 0:
                all_poses.append(json.load(open(dir + "/gt.json")))
            all_poses.append(json.load(open(dir + "/est.json")))

with open(fullpath + "output.csv", "w", newline="") as file:
    print(fullpath + "output.csv")
    writer = csv.writer(file)
    # Define column names
    column_names = [
        "name",
        "nfeat",
        "scale",
        "denoise",
        "err_avg",
        "err_std",
        "err_max",
        "steps_sec",
    ]  # Replace with your actual column names
    # Write column names as the first row
    writer.writerow(column_names)
    # Write each list in 'final' as a row in the CSV
    for row in final:
        writer.writerow(row)
map = draw_maps(all_poses)


cv.imshow("map", map)
cv.waitKey(0)
breakpoint()
