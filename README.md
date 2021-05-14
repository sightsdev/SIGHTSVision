# SIGHTSVision

This repository contains vision and camera related software created by the Semi-Autonomous Rescue Team for the SART Integrated GUI and Host Teleoperation Service (SIGHTS) project.

![image](https://www.sfxrescue.com/wp-content/uploads/2019/07/7.png)

## Requirements

### 1. Python Packages

Packages required are Imutils, Scipy, OpenCV version 3.4, Numpy and Pyzbar. Install with this command:  
```
$ python -m pip install imutils scipy numpy pyzbar opencv-contrib-python==3.4.13.47
```

### 2. ZED SDK
The spatial mapping script uses the ZED SDK to interface with ZED, a depth-sensing camera from StereoLabs.

### 3. YOLO Object Detection
This project will eventually contain Darknet binaries, which we will reference as a tool to make detections in our current program.  
Darknet is an implementation of the YOLO algorithm by AlexeyAB (https://github.com/AlexeyAB/darknet). It is among the most widely used open-source implementations of YOLO.

## Downloading and Running

The main file for hazmat detection is hazmat.py. Use the -h flag to see all the parameters.  
```
$ python hazmat.py -h
```

Use the -vs flag for the video source. w for webcam, r for robot camera.  
```
$ python hazmat.py -vs w
$ or
$ python hazmat.py -vs r
```


## Contributing

If you have an idea, suggestion or bug report for **SIGHTSVision**, or want to make a contribution of your own, we'd love to work with you to make it happen! Take a look at our [contributing page](https://github.com/SFXRescue/.github/blob/master/CONTRIBUTING.md) for more information.
