import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

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


print(args["path"])
if args["path"] != "latest":
    fullpath = args["path"] + "/output.csv"
else:
    path = os.getcwd() + "/data/output/"
    dirs = [path + d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs.sort()
    fullpath = dirs[-1]
    fullpath += "/output.csv"


# Load the CSV file
data = pd.read_csv(fullpath)

orb = data[data["fm"] == "ORB"].sort_values("steps_sec")

plt.plot(
    orb["steps_sec"],
    orb["err_avg"],
    label="Column 1",
)  # Replace 'x_column1' and 'y_column1' with the names of your columns

sift = data[data["fm"] == "SIFT"].sort_values("steps_sec")


plt.plot(
    sift["steps_sec"],
    sift["err_avg"],
    label="Column 2",
)  # Replace 'x_column2' and 'y_column2' with the names of your columns
# Add labels and title
plt.xlabel("FPS")  # Replace 'X-axis label' with the label for the x-axis
plt.ylabel("error")  # Replace 'Y-axis label' with the label for the y-axis
plt.title("Title")  # Replace 'Title' with the title of the graph

# Add a legend
plt.legend()

# Display the plot
plt.show()
