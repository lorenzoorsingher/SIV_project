import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="vo.py",
        description="""Run visual odometry on any video, 
        when no arguments are provided the script will run
        in KITTI mode on sequence 0 with SIFT_KNN.""",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["video", "kitti"],
        type=str,
        help="Set the input mode for VO. ['video'|'kitti']",
        default="kitti",
        metavar="",
    )

    parser.add_argument(
        "-s",
        "--sequence",
        type=int,
        help="Index of KITTI sequence [kitti mode]",
        default=0,
        metavar="",
    )

    parser.add_argument(
        "-st",
        "--steps",
        type=int,
        help="Number of frames to analyse. -1 for all frames.",
        default=-1,
        metavar="",
    )

    parser.add_argument(
        "-f",
        "--frameskip",
        type=int,
        help="Set the frequency of frames to analyse",
        default=1,
        metavar="",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output folder for the results",
        default="",
        metavar="",
    )

    parser.add_argument(
        "-nd",
        "--no-debug",
        action="store_true",
        help="Do not show debug windows",
        default=False,
    )

    parser.add_argument(
        "-fm",
        "--feat-match",
        type=int,
        help="Set the feature matching method. [0|1|2|3|4|5] (ORB_BF, ORB_FLANN, SIFT_FLANN, *SIFT_KNN, ORB_KNN, SIFT_FLANN_LOWE)",
        default=3,
        metavar="",
    )

    parser.add_argument(
        "-nf",
        "--num-feat",
        type=int,
        help="Set the number of features to detect in each frame. [video mode]",
        default=500,
        metavar="",
    )

    parser.add_argument(
        "-cp",
        "--calib-path",
        type=str,
        help="Path to the calibration file. [video mode]",
        default="camera_data/calib.json",
        metavar="",
    )

    parser.add_argument(
        "-vp",
        "--video-path",
        type=str,
        help="Path to the video file. [video mode]",
        default="data/video.MOV",
        metavar="",
    )

    parser.add_argument(
        "-ki",
        "--kitti-imgs",
        type=str,
        help="Path to the data_odometry_gray folder. [kitti mode]",
        default="data/data_odometry_gray",
        metavar="",
    )

    parser.add_argument(
        "-kp",
        "--kitti-poses",
        type=str,
        help="Path to the data_odometry_poses folder. [kitti mode]",
        default="data/data_odometry_poses",
        metavar="",
    )

    parser.add_argument(
        "-sf",
        "--scale-factor",
        type=float,
        help="Amount of scaling to apply to the images",
        default=1.0,
        metavar="",
    )

    parser.add_argument(
        "-de",
        "--denoise",
        type=int,
        help="Amount (size of kernel) of gaussian blur to apply to the images for denoising. Default 0",
        default=0,
        metavar="",
    )

    parser.add_argument(
        "-ms",
        "--map-size",
        type=int,
        help="Size of map",
        default=1000,
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
