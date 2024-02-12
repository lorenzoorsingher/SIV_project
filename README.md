
# CAFTP (Cool Acronym For The Project)

Project for Signal Image and Video, UniTN

## About
The goal of the project is to build a Monocular Visual Odometry system from scratch, without the assistance of machine learning or neural networks, relaying on traditional computer vision tecniques 

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
├── odometry.py                         # VOAgent and Position classes
└── requirements.txt                    # Requirements
```
