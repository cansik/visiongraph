<img src="https://github.com/user-attachments/assets/0ed34695-ca0e-47ff-aebb-eb59ff851770" alt="Visiongraph Logo Bright" width="75%">

# Visiongraph

[![PyPI](https://img.shields.io/pypi/v/visiongraph)](https://pypi.org/project/visiongraph/)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/visiongraph)
[![Documentation](https://img.shields.io/badge/read-documentation-blue)](https://cansik.github.io/visiongraph/visiongraph.html#documentation)

Visiongraph is a high level computer vision framework that includes predefined modules to quickly create and run algorithms on images. It is based on opencv and includes other computer vision frameworks like [Intel openVINO](https://github.com/openvinotoolkit/openvino) and [Google MediaPipe](https://github.com/google-ai-edge/mediapipe).

Here an example on how to start a webcam capture and display the image:

```python
from visiongraph import vg

vg.create_graph(vg.VideoCaptureInput()).then(vg.ImagePreview()).open()
```

Get started with `visiongraph` by reading the **[documentation](https://cansik.github.io/visiongraph/visiongraph.html#documentation)**.

## Installation
Visiongraph supports Python 3.10 and 3.11. Other versions may also work, but are not officially supported. Usually this is a third-party dependency problem: for example, [pyrealsense2](https://pypi.org/project/pyrealsense2/#files) does not have wheel packages for `3.12`.

To install visiongraph with all dependencies call [pip](https://pypi.org/project/pip/) like this:

```bash
pip install "visiongraph[all]"
```

It is also possible to only install certain packages depending on your needs (recommended):

```bash
# example on how to install realsense and openvino support only
pip install "visiongraph[realsense, openvino]"
```

Please read more about the extra packages in the [documentation](https://cansik.github.io/visiongraph/visiongraph.html#extras).

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
- [WebRTC Input](https://github.com/cansik/visiongraph-webrtc) - WebRTC input example for visiongraph

## Development
To develop on visiograph it is recommended to clone this repository and install the dependencies like this:

```bash
# in the visiongraph directory
pip install -e ".[all]"
```

### Build
To build a new wheel package of visiongraph run the following command in the root directory.

```bash
python setup.py bdist_wheel
```

### Docs

To generate the documentation, use the following commands.

```bash
# create documentation into "./docs
python setup.py doc

# launch pdoc webserver
python setup.py doc --launch
```

## Dependencies

Parts of these libraries are directly included and adapted to work with visiongraph.

* [motpy](https://github.com/wmuron/motpy) - simple multi object tracking library (MIT License)
* [motrackers](https://github.com/adipandas/multi-object-tracker) - Multi-object trackers in Python (MIT License)
* [OneEuroFilter-Numpy](https://github.com/HoBeom/OneEuroFilter-Numpy) - (MIT License)

Here you can find a list of the dependencies of visiongraph and their licence:

```
depthai               MIT License
faiss-cpu             MIT License
filterpy              MIT License
mediapipe             Apache License 2.0
moviepy               MIT License
numba                 BSD License
onnxruntime           MIT License
onnxruntime-directml  MIT License
onnxruntime-gpu       MIT License
opencv-python         Apache License 2.0
openvino              Apache License 2.0
pyk4a-bundle          MIT License
pyopengl              BSD License
pyrealsense2          Apache License 2.0
pyrealsense2-macosx   Apache License 2.0
requests              Apache License 2.0
scipy                 MIT License
SpoutGL               BSD License
syphon-python         MIT License
tqdm                  MIT License
vector                BSD License
vidgear               Apache License 2.0
wheel                 MIT License
```

For more information about the dependencies have a look at the [requirements.txt](https://github.com/cansik/visiongraph/blob/main/requirements.txt).

Please note that some models (such as Ultralytics YOLOv8 and YOLOv11) have specific licences (AGPLv3). Always check the model licence before using the model.

## About
Copyright (c) 2024 Florian Bruggisser