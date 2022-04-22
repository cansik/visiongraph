# Visiongraph [![PyPI](https://img.shields.io/pypi/v/visiongraph)](https://pypi.org/project/visiongraph/)
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
- [InpaintExample](examples/InpaintExample.py) - GAN based inpainting example.
- [MidasDepthExample](examples/MidasDepthExample.py) - Realtime depth prediction with the [midas-small](https://github.com/isl-org/MiDaS) network.
- [RGBDSmoother](examples/RGBDSmoother.py) - Smooth RGB-D depth map videos with a one-euro filter per pixel.

There are even more examples where visiongraph is currently in use:

- [Spout/Syphon RGB-D Example](https://github.com/cansik/spout-rgbd-example) - Share RGB-D images over spout or syphon.

## Documentation
This documentation is intended to provide an overview of the framework. A full documentation will be available later.

### Graph
The core component of visiongraph is the [BaseGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/BaseGraph.py) class. It contains and handles all the nodes of the graph. A BaseGraph can run on the same thread as called or a new thread or process. The nodes in the graph are just a list, the graph itself is created by nesting nodes into each other.

#### Graph Node
A [GraphNode](https://github.com/cansik/visiongraph/blob/main/visiongraph/GraphNode.py) is a single step in the graph. It has a input and output type and processes the data within the `process()` method.

#### Graph Builder
The graph builder helps to create new graphs on a single line in python. It creates a [VisionGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/VisionGraph.py) object which is a child of the BaseGraph. The following code snippet is an example of the graph builder which creates a smooth pose estimation graph.

```python
import visiongraph as vg

graph = vg.create_graph(name="Smooth Pose Estimation",
                            input_node=vg.VideoCaptureInput(0),
                            handle_signals=True) \
        .apply(ssd=vg.sequence(vg.OpenPoseEstimator.create(), vg.MotpyTracker(), vg.LandmarkSmoothFilter()),
               image=vg.passthrough()) \
        .then(vg.ResultAnnotator(image="image"), vg.ImagePreview()) \
        .open()
```

### Input
Supported are image, video, webcam, RealSense and Azure Kinect input types.

### Estimator
Usually an estimator is a graph node which takes an image as an input and estimates an information about the content. This could be a pose estimation or a face detection. It is also possible to have a transformation of the image, for example de-blurring it or estimate the depth map.

### Object Detection Tracker
Object detection trackers allow a detected object to be assigned an id that remains the same across successive frames.

### DSP (Digital Signal Processing)
To filter noisy estimations or inputs, the DSP package provides different filters which can be applied directly into a graph.

### Recorder
To record incoming frames or annotated results, multiple frame recorders are provided.

### Assets
Most estimators use big model and weight descriptions for their neural networks. To keep visiongraph small and easy to install, these assets are hosted externally on github. Visiongraph provides a system to directly download and cache these files.

### Argparse
To support rapid prototyping many graph and estimator options are already provided to add to the argparse parser.

## Roadmap
Next roadmap points:

- Async input and network model (run when ready)

## About
Copyright (c) 2022 Florian Bruggisser

### Included Libraries

Parts of these libraries are directly included and adapted to work with visiongraph.

* [motpy](https://github.com/wmuron/motpy) - simple multi object tracking library (MIT License)
* [motrackers](https://github.com/adipandas/multi-object-tracker) - Multi-object trackers in Python (MIT License)
* [OneEuroFilter-Numpy](https://github.com/HoBeom/OneEuroFilter-Numpy) - (MIT License)

For more information about the dependencies have a look at the [requirements.txt](https://github.com/cansik/visiongraph/blob/main/requirements.txt).