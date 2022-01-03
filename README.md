# Visiongraph
Visiongraph is a high level computer vision pipeline that includes predefined modules to quickly create and run algorithms on images. It is based on opencv and includes other computer vision frameworks like [Intel openVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) and [Google MediaPipe](https://google.github.io/mediapipe/).

The main goal is to implement a platform independent and high performance framework for day-to-day computer vision tasks.

## Installation
To install visiongraph with all dependencies (which are cross-platform) call [pip](https://pypi.org/project/pip/) like this:

```bash
pip install "visiongraph[all]"
```

🚨 *Please note that visiongraph is in an early alpha phase and the API will still undergo changes.*

It is also possible to only install certain packages depending on your needs:

```bash
pip install "visiongraph[realsense, openvino, mediapipe, onnx, media, azure]"
```

#### Azure Kinect
The Azure Kinect is only available on Linux & Windows systems and will not be installed with `all` extra identifier. To install the Azure Kinect support, specify the azure kinect as well:

```bash
pip install "visiongraph[all, azure]"
```

#### RealSense Camera on MacOS
Because Intel does not release the Intel librealsense2 python bindings for MacOS ([#9687](https://github.com/IntelRealSense/librealsense/issues/9687)), you have first to install it from pre-built repository like [pyrealsense2-macosx](https://github.com/cansik/pyrealsense2-macosx). It is recommended to use pip with the [find-links](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages) command.

```bash
pip install pyrealsense2 --find-links https://github.com/cansik/pyrealsense2-macosx/releases
```

### Development
To develop visiograph itself it is recommended to clone this repository and install the dependencies like this:

```bash
# in the visiongraph directory
pip install -e ".[all]"
```

### Build
To build a new wheel package of visiongraph run the following command in the root directory.

```bash
python setup.py bdist_wheel
```

## Examples
To demonstrate the possibilities of visiongraph there are already implemented [examples](examples) ready for you to try out. Here is a list of the current examples:

- [InputExample](examples/InputExample.py) - A basic input example that determines the center if possible.
- [RealSenseDepthExample](examples/RealSenseDepthExample.py) - Display the RealSense depth map.
- [FaceDetectionExample](examples/FaceDetectionExample.py) - A face detection pipeline example.
- [CascadeFaceDetectionExample](examples/CascadeFaceDetectionExample.py) -  A face detection pipeline that also predicts other feature points of the face.
- [HandDetectionExample](examples/HandDetectionExample.py) - A hand detection pipeline example.
- [PoseEstimationExample](examples/PoseEstimationExample.py) - A pose estimation pipeline which annotates the generic pose keypoints.
- [ProjectedPoseExample](examples/ProjectedPoseExample.py) -  Project the pose estimation into 3d space with the RealSense camera.
- [ObjectDetectionExample](examples/ObjectDetectionExample.py) - An object detection & tracking example.
- [MidasDepthExample](examples/MidasDepthExample.py) - Realtime depth prediction with the [midas-small](https://github.com/isl-org/MiDaS) network.
- [RGBDSmoother](examples/RGBDSmoother.py) - Smooth RGB-D depth map videos with a one-euro filter per pixel.

There are even more examples where visiongraph is currently in use:

- [Spout RGB-D Example](https://github.com/cansik/spout-rgbd-example) - Share RealSense RGB-D images over spout.

## Documentation

### Pipeline

### Input
Supported are video, webcam, RealSense and Azure Kinect input types. Azure Kinect may need a [special install](https://github.com/etiennedub/pyk4a#windows) on Windows and the [Azure-Kinect-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)

### Estimator

### Tracker

### DSP

### Recorder

### Assets

### Argparse

## About
Copyright (c) 2022 Florian Bruggisser