
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

### with KITTI
Download and extract <a href=https://www.cvlibs.net/datasets/kitti/eval_odometry.php>KITTI visual odometry</a> datasets in the <i>data</i> folder

The full 22GB dataset with all 22 sequences is aviable at this link: <a href=https://www.cvlibs.net/datasets/kitti/user_login.php>KITTI image data</a>. 

A lighter 225MB version containig only sequence 22 is available at this link on <a href="https://drive.google.com/drive/folders/10kYdoqBTExbuCfZMXPpRT6pO4E5a42TR?usp=drive_link">Google Drive</a>.

```
data/data_odometry_gray
```

In main.py set <i>SEQUENCE</i> to the number of the sequence you want to run. Keep in mind the light version of the dataset only contains sequence number 22 which is NOT included in the full dataset.

```
SEQUENCE = 22
```

Run main.py

```
python main.py
```


## TODO 

- add some default test data
- loop closure
- comparative testing sys
- you tell me


## Authors 

- [@lorenzoorsingher](https://github.com/lorenzoorsingher)
- [@iiics](https://github.com/iiics)