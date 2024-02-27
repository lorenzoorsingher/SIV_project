import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="vo.py",
        description="Run visual odometry on any video",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["video", "kitty"],
        type=str,
        help="Set the input mode for VO. ['video'|'kitti']",
        default="kitti",
        metavar="",
    )

    parser.add_argument(
        "-s",
        "--sequence",
        type=int,
        help="Index of KITTI sequence",
        default=0,
        metavar="",
    )

    parser.add_argument(
        "-st",
        "--steps",
        type=int,
        help="Number of frames to analize. -1 for all frames.",
        default=-1,
        metavar="",
    )

    parser.add_argument(
        "-f",
        "--frameskip",
        type=int,
        help="Set the frequency of frames to analize",
        default=1,
        metavar="",
    )

    parser.add_argument(
        "-nd",
        "--no-debug",
        action="store_false",
        help="Do not show debug windows",
        default=False,
    )

    parser.add_argument(
        "-fm",
        "--feature-matcher",
        type=int,
        help="Set the feature matcher method. [0|1|2|3|4|5] (ORB_BF, ORB_FLANN, SIFT_FLANN, *SIFT_KNN, ORB_KNN, SIFT_FLANN_LOWE)",
        default=3,
        metavar="",
    )

    args = vars(parser.parse_args())

    return args
