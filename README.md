# SIGHTSVision
This repository contains vision and camera related software created by the Semi-Autonomous Rescue Team for the SART Integrated GUI and Host Teleoperation Service (SIGHTS) project


## Requirements

* Imutils: ```python -m pip install imutils```
* Scipy: ```python -m pip install scipy```
* OpenCV (version 3.4): ```pip install opencv-python==3.4.13.47```
* Numpy: ```python -m pip install numpy```


## Downloading and Running

The main file for hazmat detection is hazmat.py. Use the -h flag to see all the available parameters:
```python hazmat.py -h```

Use the -vs flag for the video source. w for webcam, r for robot camera:
* ```python hazmat.py -vs w```
* ```python hazmat.py -vs r```


## Contributing

If you have an idea, suggestion or bug report for **SIGHTSVision**, or want to make a contribution of your own, we'd love to work with you to make it happen! Take a look at our [contributing page](https://github.com/SFXRescue/.github/blob/master/CONTRIBUTING.md) for more information.
