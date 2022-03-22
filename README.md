# Visiongraph
Visiongraph is a high level computer vision pipeline that includes predefined modules to quickly create and run algorithms on images. It is based on opencv and includes other computer vision frameworks like [Intel openVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) and [Google MediaPipe](https://google.github.io/mediapipe/).

Here an example on how to start a webcam capture and display the image:

```python
import visiongraph as vg
vg.create_graph(vg.VideoCaptureInput()).then(vg.ImagePreview()).open()
```

The main goal is to implement a platform independent and high performance framework for day-to-day computer vision tasks.

## Installation
To install visiongraph with all dependencies call [pip](https://pypi.org/project/pip/) like this:

```bash
pip install "visiongraph[all]"
```

🚨 *Please note that visiongraph is in an early alpha phase and the API will still undergo changes.*

It is also possible to only install certain packages depending on your needs:

```bash
pip install "visiongraph[realsense, openvino, mediapipe, onnx, media, azure, numba]"
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

- [SimpleVisionGraph](examples/SimpleVisionGraph.py) - SSD object detection & tracking of live webcam input with `5` lines of code.
- [VisionGraphExample](examples/VisionGraphExample.py) - A face detection and tracking example with custom events.
- [InputExample](examples/InputExample.py) - A basic input example that determines the center if possible.
- [RealSenseDepthExample](examples/DepthCameraExample.py) - Display the RealSense or Azure Kinect depth map.
- [FaceDetectionExample](examples/FaceDetectionExample.py) - A face detection pipeline example.
- [FindFaceExample](examples/FindFaceExample.py) - A face recognition example to find a target face.
- [CascadeFaceDetectionExample](examples/CascadeFaceDetectionExample.py) -  A face detection pipeline that also predicts other feature points of the face.
- [HandDetectionExample](examples/HandDetectionExample.py) - A hand detection pipeline example.
- [PoseEstimationExample](examples/PoseEstimationExample.py) - A pose estimation pipeline which annotates the generic pose keypoints.
- [ProjectedPoseExample](examples/ProjectedPoseExample.py) -  Project the pose estimation into 3d space with the RealSense camera.
- [ObjectDetectionExample](examples/ObjectDetectionExample.py) - An object detection & tracking example.
- [InstanceSegmentationExample](examples/InstanceSegmentationExample.py) - Intance Segmentation based on COCO80 dataset.
- [MidasDepthExample](examples/MidasDepthExample.py) - Realtime depth prediction with the [midas-small](https://github.com/isl-org/MiDaS) network.
- [RGBDSmoother](examples/RGBDSmoother.py) - Smooth RGB-D depth map videos with a one-euro filter per pixel.

There are even more examples where visiongraph is currently in use:

- [Spout/Syphon RGB-D Example](https://github.com/cansik/spout-rgbd-example) - Share RGB-D images over spout or syphon.

## Documentation

This documentation is intended to provide an overview of the framework. A full documentation will be available later.

### Graph

The core component of visiongraph is the [BaseGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/BaseGraph.py) class.

#### GraphNode

#### Graph Builder

### Input
Supported are video, webcam, RealSense and Azure Kinect input types.

### Estimator

### Tracker

### DSP

### Recorder

### Assets

### Argparse

##Roadmap
Next roadmap points:

- Async input and network model (run when ready)

## About
Copyright (c) 2022 Florian Bruggisser

### Included Libraries

Parts of these libraries are directly included and adapted to work with visiongraph.

* [motpy](https://github.com/wmuron/motpy) - simple multi object tracking library (MIT License)
* [motrackers](https://github.com/adipandas/multi-object-tracker) - Multi-object trackers in Python (MIT License)
* [OneEuroFilter-Numpy](https://github.com/HoBeom/OneEuroFilter-Numpy) - (MIT License)