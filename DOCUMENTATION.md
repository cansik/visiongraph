# Documentation

This documentation is intended to provide an overview of the framework.

## Getting Started

### Structure

The visiongraph package structure contains the following main packages.

- `vg` - Supports lazy and optional access to all modules via a single package import.
- `input` - Contains camera support for various cameras (like UVC, Azure Kinect, RealSense and so on).
- `estimator` - Implements the machine learning models and computer vision algorithms.
- `result` - Contains the models for the estimator results.
- `output` - Adds output support like NDI, Syphon, Spout and image preview.

Additionally, there are the following packages.

- `model` - Contains class models that are necessary for visiongraph.
- `dsp` - Contains filters and DSP algorithms.
- `recorder` - Adds support for video recordings using various frameworks.
- `tracker` - Contains object-detection tracker implementations.

### Import Visiongraph

There are two ways on how to import visiongraph related objects and classes. The classical way is to use the direct
import like this:

```python
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine

engine = OpenVinoEngine(...)
```

However, due to the number of packages and package depth in visiongraph, it is recommended to use the `vg` package,
which includes all public modules and methods. This code is optimized to lazily import the requested module when called
the first time.

```python
from visiongraph import vg

engine = vg.OpenVinoEngine(...)
```

#### Optional Imports

`vg` allows for direct access of all public members of visiongraph and even handles optional imports. If an import is
not available, a stub-object is returned which throws an error on accessing its attributes. The reason behind this is,
that it is possible to work with types, which would not be accessible on certain operating systems (like macOS):

```python
from visiongraph import vg

device = ...

if isinstance(device, vg.AzureKinectInput):
    # would always be "False" on MacOS
    print("This is a Kinect")
```

### Graph Node

A `visiongraph.GraphNode.GraphNode` is a single processable unit and usually only solves one specific task. However, the
philosophy of visiongraph is to have rather a few rich nodes, instead of every small task exposed as a single node. Each
node implements the `visiongraph.Processable.Processable` interface, which has a input and output type and processes the
data within the `process(self, data: InputType) -> OutputType` method.

Since a lot of algorithms and node implementations need to acquire and release resources (like camera handles, gpu
memory, native frameworks), it adds lifecycle methods to `setup()` and `release()` a node. The usualy way to use a
`visiongraph.GraphNode.GraphNode` is to create an instance, call `setup()`, use it and `release()` it again.

Here is an example of a basic pose estimator:

```python
from visiongraph import vg

# create the instance of a pose estimator
mp_pose = vg.MediaPipePoseEstimator()

# prepare the necessary resources and start the estimator
mp_pose.setup()

# process a frame
result = mp_pose.process(my_np_image)

# release the resources
mp_pose.release()
```

#### Context Manager

Usually, the `setup()` and `release()` methods should not be used, each node implements the context-manager pattern and
can be used like this (to avoid not cleaning up the resources).

```python
from visiongraph import vg

with vg.MediaPipePoseEstimator() as mp_pose:
    # process a frame
    result = mp_pose.process(my_np_image)
```

## Input

Since there are a lot of special cameras which support different capture modes, visiongraph implements a basic
abstraction for various computer-vision cameras like UVC-webcams, RealSense, Azure Kinect, ZED, OAK, etc. The
abstraction is called `visiongraph.input.BaseInput.BaseInput` and allows to `read()` from a specific device or stream.
The input usually is a camera but can also be an NDI, WebRTC or basic video stream.

Here is an example on how to read frames from a camera. Be aware that `read()` returns a `timestamp` in milliseconds and
an optional camera image in `np.ndarray` format (usually as HWC and BGR).

```python
from visiongraph import vg

# create the camera instance but does not connect to the hardware
cam = vg.VideoCaptureInput()

# connects to the actual camera hardware
cam.setup()

# using the camera to read frames
running = True
while running:
    ts, frame = cam.read()
    if frame is None:
        running = False
        continue

    # do something with the frame

# release the camera handle
cam.release()
```

It is also possible to apply post-processing methods like `rotate`, `flip`, `mask`  or `crop` by configuring them on the
`visiongraph.input.BaseInput.BaseInput`.

```python
from visiongraph import vg
import cv2

cam = vg.VideoCaptureInput()
cam.rotate = cv2.ROTATE_90_CLOCKWISE
cam.setup()
```

### Depth Camera

Since in computer vision special cameras with multiple streams are more common, there is the
`visiongraph.input.BaseDepthCamera.BaseDepthCamera` abstraction, which abstracts cameras providing color, infrared and
depth streams like the RealSense, Azure Kinect or Luxonis OAK cameras. The `BaseDepthCamera` adds support to read frames
by stream, either in raw format (as it is captured from the camera) or in a pre-processed format (which is a unified
format for all depth cameras).

```python
from visiongraph import vg

cam: vg.BaseDepthCamera = vg.AzureKinectInput()

# setup settings for the depth camera (like enabling the infrared and depth stream)
cam.use_infrared = True
cam.enable_depth = True

cam.setup()

# read() still returns just a single ts and frame (usually the default color stream)
ts, frame = cam.read()

# to read a depth frame, use the following methods
depth_image = cam.depth_image  # colorized np.uint8 frame (for preview)
raw_depth_image = cam.raw_depth_image  # raw np.uint16 (for azure kinect) for processing
```

It is also possible to dynamically read a specific frame by using the
`visiongraph.input.BaseDepthCamera.BaseDepthCamera.get_image()` method. In this context, `pre_processed` refers to the
image processing that happens directly by the specific depth camera implementation (for example, the `AzureKinectInput`
normalizes the infrared image to a min/max value). `post_processed` is a flag to enable the `rotate`, `flip` and other
post-processing methods by the `visiongraph.input.BaseInput.BaseInput`.

```python
from visiongraph import vg

with vg.AzureKinectInput() as cam:
    raw_infrared = cam.get_image(vg.CameraStreamType.Infrared, pre_processed=False, post_processed=False)
```

### Camera Intrinsics

For many computer vision applications the camera intrinsics are an essential information. The
`visiongraph.input.BaseCamera.BaseCamera` adds methods that camera implementations can use to return camera intrinsics
per stream type (by default the color).

```python
from visiongraph import vg

with vg.AzureKinectInput() as cam:
    infrared_intrinsics = cam.get_intrinsics(vg.CameraStreamType.Infrared)

    print(infrared_intrinsics.camera_matrix)
    print(infrared_intrinsics.distortion_coefficients)

    principle_point = infrared_intrinsics.px, infrared_intrinsics.py
    focal_point = infrared_intrinsics.fx, infrared_intrinsics.fy
```

### Settings

Usually, the input is configured between the object initialisation and the call of the `setup()` method. However, it is
also possible to control the camera parameters at runtime. A very basic subset of controls to change `gain`, `exposure`
and `white-balance` is implemented into the `visiongraph.input.BaseCamera.BaseCamera`.

```python
from visiongraph import vg

with vg.AzureKinectInput() as cam:
    cam.enable_auto_exposure = True
    cam.gain = 100
```

## Estimator

An estimator is typically a graph node that takes an image as input and produces information about its content. Examples
include pose estimation or face detection. In some cases, the estimator may also transform the image itself, such as by
removing blur or generating a depth map.

### Object Detection

![pexels-jimbear-2926723-crowdhuman.jpg](doc/pexels-jimbear-2926723-crowdhuman.jpg)

There are various implementations of object detectors in visiongraph, spanning from [SSD](https://arxiv.org/abs/1512.02325), [YOLO](https://arxiv.org/abs/1506.02640) (X, v5, v8, v8OBB, etc.)
over face detectors to specific human crowd detectors. Each object detector returns a list of results which can be used
to extract further information about the object instance.

```python
import numpy as np

from visiongraph import vg

face_image: np.ndarray

with vg.MediaPipeFaceDetector() as face_detector:
    results: vg.ResultList[vg.FaceDetectionResult] = face_detector.process(face_image)

    for face in results:
        print(face.class_id)
        print(face.class_name)
        print(face.score)
        print(face.bounding_box)

    # simply annotate the faces on the image
    results.annotate(face_image)
```

Since machine learning frameworks usually need specific model and weight descriptions, visiongraph already provides a
list of configurations per detector. These configurations are only available, if the model and weights are already
hosted in the repository (see [assets](#Assets)). Here are some examples:

```python
from visiongraph import vg

# face detector with config
with vg.AdasFaceDetector.create(vg.AdasFaceConfig.MobileNet_672x384_FP32) as face_detector:
    pass

# generic YOLOv8 object detector with config
with vg.YOLOv8Detector.create(vg.YOLOv8Config.YOLOv8_N) as detector:
    pass

# mask RCNN estimator for instance segmentation tasks
with vg.MaskRCNNEstimator.create(vg.MaskRCNNConfig.EfficientNet_480_INT8) as segmenter:
    pass
```

#### Non-Maximum-Suppression ([NMS](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/))

Most of the object detection models need a post non-maximum-suppression to remove bounding boxes that have been annotated more than once.

```python
results = [vg.ObjectDetectionResult(...), ...]
vg.non_maximum_suppression(results, batched=True)
```

It is possible to either run NMS over all object detection results, or make it class-aware (`batched`). By default, it is not class-aware. Usually, estimators allow to pass a `visiongraph.model.NMSOptions.NMSOptions` to configure the internal NMS call.

### Human Pose Estimator

Human pose estimation is a task very common in interactive systems and is basically a subtask of object detection (on a result level). That is why visiongraph models the human pose estimation task as a `visiongraph.estimator.spatial.ObjectDetector.ObjectDetector` but adds additional methods to work with landmarks (keypoints).

```python
from visiongraph import vg

pose_image: np.ndarray

with vg.AEPoseEstimator.create() as pose_detector:
    results = pose_detector.process(pose_image)
    for pose in results:
        eye = pose.left_eye
        print(f"Eye Position: {eye.x}, {eye.y}")
        print(f"Landmark count: {len(pose.landmarks)}")
```

There are various models implemented, usually it's recommended to use the mediapipe pose model which is very efficient. Here you find a selection of models.

```python
from visiongraph import vg

vg.MediaPipePoseEstimator.create(vg.MediaPipePoseConfig.Light)
vg.AEPoseEstimator.create(vg.AEPoseConfig.EfficientHRNet_288_FP16)
vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_S_COCO_640)
vg.UltralyticsPoseEstimator.create(vg.UltralyticsPoseConfig.YOLOv8_N_640_INT8)
vg.LiteHRNetPoseEstimator.create(vg.LiteHRNetConfig.LiteHRNet_30_COCO_384x288_FP16)
vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_MultiPose_320x320_FP32)
vg.OpenPoseEstimator.create(vg.OpenPoseConfig.LightWeightOpenPose_INT8)
```

Since human pose datasets are not always use the same amount of landmarks, there is a generic `visiongraph.result.spatial.pose.PoseLandmarkResult.PoseLandmarkResult` which allows to access default landmarks, available in all pose definitions (COCO, BlazePose, OpenPose, etc.).

```python
from visiongraph import vg

result: vg.COCOOpenPose

result.nose
result.neck
result.left_knee
result.right_knee

# and so on
```

### Hand Estimator

Similar to the human pose estimators, there are pose estimators for the hand pose detection task. It returns a list of landmarks of a hand and can be used in combination with the human pose estimation (holistic human pose detection).

```python
from visiongraph import vg

image: np.ndarray

with vg.MediaPipeHandEstimator() as network:
    results = network.process(image)

    for hand in results:
        if hand.handedness == vg.Handedness.LEFT:
            print(hand.index_finger_ip)
```

### Landmark Embeddings

For classification or re-identification tasks it can be necessary to embed the landmark result into a position and scale invariant format (normalisation). For that purpose there is already a pre-defined `visiongraph.estimator.embedding.LandmarkEmbedder.LandmarkEmbedder` which requires an embedding method and a list of pose results.

```python
from visiongraph import vg
poses: list[vg.PoseLandmarkResult] = []

with vg.LandmarkEmbedder(vg.embed_pose) as model:
    embeddings = model.process(poses)
```

### Object Segmentation

Object segmentation estimators not only predict the bounding box around an object, but predict a pixel-based mask to define the visible shape of the object. `visiongraph.estimator.spatial.InstanceSegmentationEstimator.InstanceSegmentationEstimator` inherits ` visiongraph.estimator.spatial.ObjectDetector.ObjectDetector` and extends the object detection results with a binary mask.

```python
from visiongraph import vg

image: np.ndarray

with vg.ModNetEstimator.create() as model:
    results = model.process(image)

    for instance in results:
        mask: np.ndarray = instance.mask
```

Please be aware that the mask is a binary mask (containing only 0 or 1) and is of type `np.uint8`. The size of the mask is the same as the input frame. This can be quite memory intense depending on the amount of instances that are detected.


### Camera Pose Estimator

At the moment there are only a few tools implemented for camera calibration and pose estimation. One of them is the `visiongraph.estimator.spatial.camera.ArUcoCameraPoseEstimator.ArUcoCameraPoseEstimator` which requires camera intrinsics to detect ArUco markers and predict the relative camera pose.

```python
from visiongraph import vg

intrinsics = vg.CameraIntrinsics.load("media/calibration.json")
image: np.ndarray

with vg.ArUcoCameraPoseEstimator(
            camera_matrix=intrinsics.intrinsic_matrix, fisheye_distortion=intrinsics.distortion_coefficients
        ) as model:
    
    pose = model.process(image)
    print(f"Camera position: {pose.position} / rotation {pose.rotation}")
    print(f"Distance: {pose.position.mag:.2f}")
```

For creating an intrinsic camera calibration, have a look at `visiongraph.estimator.spatial.camera.ChessboardCalibrator.ChessboardCalibrator` or `visiongraph.estimator.spatial.camera.ChArUcoCalibrator.ChArUcoCalibrator`.

### Result

### Inference Engine

## Object Detection Tracker

Object detection trackers allow a detected object to be assigned an `tracking_id` that remains the same across successive frames.

## DSP (Digital Signal Processing)

To filter noisy estimations or inputs, the DSP package provides different filters which can be applied directly into a
graph.

## Recorder

To record incoming frames or annotated results, multiple frame recorders are provided.

## Assets

Most estimators use big model and weight descriptions for their neural networks. To keep visiongraph small and easy to
install, these assets are hosted externally on github. Visiongraph provides a system to directly download and cache
these files.

## Utilities

### Argparse

To support rapid prototyping, many graph and estimator options are already provided to add to the argparse parser.
Please have a look at [Logging](#Logging) or [Input](#Input).

### Logging

To enable logging for visiongraph during the import phase, please set the following environment variable:

```bash
# zsh / bash
export VISIONGRAPH_LOGLEVEL=INFO

# cmd
set VISIONGRAPH_LOGLEVEL=INFO

# powershell
$env:VISIONGRAPH_LOGLEVEL="INFO"
```

To add loglevel argparse support, it is possible to use the following methods.

```python
from visiongraph import vg
import argparse

parser = argparse.ArgumentParser()
# add logging parameter to the parser
vg.add_logging_parameter(parser)
args = parser.parse_args()

# setup loglevel
vg.setup_logging(args.loglevel)
```

## Graph

The core component of visiongraph is
the [BaseGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/BaseGraph.py) class. It contains and handles
all the nodes of the graph. A BaseGraph can run on the same thread as called or a new thread or process. The nodes in
the graph are just a list, the graph itself is created by nesting nodes into each other.

#### Graph Builder

The graph builder helps to create new graphs on a single line in python. It creates
a [VisionGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/VisionGraph.py) object which is a child of
the BaseGraph. The following code snippet is an example of the graph builder which creates a smooth pose estimation
graph.

```python
from visiongraph import vg

graph = (
    vg.create_graph(name="Smooth Pose Estimation",
                    input_node=vg.VideoCaptureInput(0),
                    handle_signals=True)
    .apply(ssd=vg.sequence(vg.OpenPoseEstimator.create(), vg.MotpyTracker(), vg.LandmarkSmoothFilter()),
           image=vg.passthrough())
    .then(vg.ResultAnnotator(image="image"), vg.ImagePreview())
)
graph.open()
```

## Extras

It is possible to install extra module to visiongraph by specifying them when installing visiongraph. Here is a list of
currently supported extras:

- `realsense` - Support for Intel RealSense cameras
- `azure` - Support for Microsoft Azure Kinect cameras
- `depthai` - Support for the Luxonis cameras
- `openvino` - Support for the Intel openVINO machine learning framework
- `mediapipe` - Support for the Google MediaPipe machine learning framework
- `onnxruntime` - Support for the ONNX machine learning framework (CPU)
- `onnxruntime-gpu` - Support for the ONNX machine learning framework (CUDA GPU)
- `onnxruntime-directml` - Support for the ONNX machine learning framework (DirctML GPU)
- `media` - Support for VidGear and MoviePy video reading and writing
- `numba` - Improved performance for smoothing and tracking algorithms
- `fbs` - Support for framebuffer sharing (SpoutGL or Syphon)
- `faiss` - Support for fast pose classification
- `mot` - Support for multi-object-tracking using motpy
