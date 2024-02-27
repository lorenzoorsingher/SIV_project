import argparse
import pdb

parser = argparse.ArgumentParser(
    prog="VisualOdometry",
    usage="",
    description="Run visual odometry on any video",
)

parser.add_argument("-s", "--sequence")
parser.add_argument("-d", "--debug", action="store_true")


args = parser.parse_args()
print(args.filename, args.count, args.verbose)
