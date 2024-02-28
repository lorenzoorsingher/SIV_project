
# CAFTP (Cool Acronym For The Project)

Project for Signal Image and Video, UniTN

## About
The goal of the project is to build a Monocular Visual Odometry system from scratch, without the assistance of machine learning or neural networks, relaying only traditional computer vision tecniques. 
The main objective is to construct a robust system that can be used on different kinds of cameras by providing an easy to use pipeline that goes from calibration to exploration.

## General Information

The project is entirely built in python and, while for many tasks we relayed on third party libraries such as OpenCV, we made sure to implements as many components as possible from scratch, deep diving into the logic that goes behind a full VO system. For our benchmarks we used [KITTI](https://www.cvlibs.net/datasets/kitti/) as well as clips recorded by the team.

### Project Structure

```
SIV_project/                            # main project directory
├── camera_data                         # calibration files 
│   ├── calib.json
│   └── ...
├── data                                # datasets and videos
│   ├── data_odometry_gray              # KITTI sequences
│   │   └── dataset
│   │       └── sequences
│   │           ├── 00
│   │           │   ├── image_0
│   │           │   ├── image_1
│   │           │   ├── calib.txt
│   │           │   └── times.txt
│   │           └──...
│   ├── data_odometry_poses             # KITTI poses for sequences
│   │   └── dataset
│   │       └── poses
│   │           ├── 00.txt
│   │           └──  ...
│   └──  ...
├── camera_calibration_charuco.py       # Calib script for ChArUco board
├── camera_calibration_normal.py        # Calib script for standard board
├── common.py                           # Constants and common func
├── kitti_loader.py                     # KITTI loader class
├── main.py                             # main
├── VOAgent.py                          # VOAgent class
├── position.py                         # Position classe
└── requirements.txt                    # Requirements
```

## Usage

install requirements

```
pip install -r requirements.txt
```

### With KITTI
Download and extract <a href=https://www.cvlibs.net/datasets/kitti/eval_odometry.php>KITTI visual odometry</a> datasets in the <i>data</i> folder

The full 22GB dataset with all 22 sequences is aviable at this link: <a href=https://www.cvlibs.net/datasets/kitti/user_login.php>KITTI image data</a>. 

A lighter 225MB version containig only sequence 22 is available at this link on <a href="https://drive.google.com/drive/folders/1lh0QLIo15Rr3JK6u5jNcgM2uX-k4pOK9?usp=sharing">Google Drive</a>.

```
data/data_odometry_gray
```
Run vo.py in kitty mode (default)

```
python vo.py -m kitti
```

You can provide the desired <i>sequence</i> index to be run 

```
python vo.py -m kitti -s 2
```

### With videos

To run in video mode a camera calibration file must be provieded. The camera calibration file can be obtained using <i>camera_calibration_charuco.py</i> and the ChAruCo pattern inside the camera_data folder

```
python vo.py -m video -cp {calib/file/path.json} -vp {video/path.mp4}
```


### General Controls

To the script can be provided the number of frames to analyze (default -1 means the entire length)

```
python vo.py -m kitti -s 2 -st 100
```
As well as the feature matching method used as combinations of SIFT, ORB and either bruteforce or FLANN plus some more

```
python vo.py -m kitti -s 2 -fm 2
```

And the amount of scaling to apply to the images

```
python vo.py -m kitti -sf 0.5
```

## TODO 

- add some default test data
- loop closure
- comparative testing sys
- parametrized script 
- implementation of error function 
- pre-processing? (downsampling to make it faster, blurring (against noise -> more points), ...)
- report
- provide instructions for camera calibration


## Authors 

- [@lorenzoorsingher](https://github.com/lorenzoorsingher)
- [@iiics](https://github.com/iiics)