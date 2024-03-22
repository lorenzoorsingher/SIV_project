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
    in KITTI mode on sequence 0 with SIFT_BF_LOWE.""",
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
squences = []
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

    if settings["sequence"] not in squences:
        squences.append(settings["sequence"])

final = []
for key, value in aggregate.items():
    aggregate[key]["error_avg"] = np.average(value["errors"]).round(3)
    aggregate[key]["error_std"] = np.std(value["errors"]).round(3)
    aggregate[key]["error_max"] = np.max(value["errors"]).round(3)
    aggregate[key]["steps_sec"] = int(np.average(value["steps_sec"]))
    aggregate[key]["frame_time"] = (1 / np.average(value["steps_sec"])).round(3)

    final.append(
        [
            key,
            key.split("_")[0],
            "_".join(key.split("_")[1:-3]),
            key.split("_")[-3],
            key.split("_")[-2],
            key.split("_")[-1],
            aggregate[key]["error_avg"],
            aggregate[key]["error_std"],
            aggregate[key]["error_max"],
            aggregate[key]["steps_sec"],
            aggregate[key]["frame_time"],
        ]
    )

sorted = sorted(final, key=lambda x: x[6])
tops = [x[0] for x in sorted]

for s in sorted:
    print(s)


all_sequences = []

for seqid in squences:
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
    all_sequences.append(all_poses)

with open(fullpath + "output.csv", "w", newline="") as file:
    print(fullpath + "output.csv")
    writer = csv.writer(file)
    # Define column names
    column_names = [
        "name",
        "fm",
        "matcher",
        "nfeat",
        "scale",
        "denoise",
        "err_avg",
        "err_std",
        "err_max",
        "steps_sec",
        "frame_time",
    ]  # Replace with your actual column names
    # Write column names as the first row
    writer.writerow(column_names)
    # Write each list in 'final' as a row in the CSV
    for row in final:
        writer.writerow(row)


maps = []
for all_poses in all_sequences:
    map = draw_maps(all_poses)
    height = 300
    width = (height / map.shape[0]) * map.shape[1]
    map = cv.resize(map, (int(width), height))
    maps.append(map)


cv.namedWindow("map", cv.WINDOW_NORMAL)
cv.imshow("map", np.hstack(maps))
cv.waitKey(0)
