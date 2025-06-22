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

### Optional Mediapipe Support

Visiongraph can integrate Google’s [MediaPipe](https://github.com/google/mediapipe) for advanced hand, face and object tracking pipelines. Unfortunately, the official PyPI MediaPipe wheels declare a strict dependency on `numpy<2.0`, which prevents installation alongside NumPy 2.x, even though most functionality works fine with NumPy 2.0 and above. To work around this limitation, we maintain a custom [mediapipe-numpy2](https://github.com/cansik/mediapipe-numpy2) build that removes the `<2.0` pin.

When you install with the `mediapipe` extra, pip will automatically fetch the matching patched wheel for your OS and Python version.

#### Alternative: Use the Official MediaPipe Release

If you’re happy to stick with NumPy <2.0, you can skip our custom package entirely and install the upstream MediaPipe wheel from PyPI:

```bash
pip install visiongraph mediapipe
```

This will install Visiongraph plus the official `mediapipe` package (which requires `numpy<2.0`). Just make sure your environment’s NumPy version is below 2.0 when using this route.


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
- [FaceMeshVVADExample.py](examples/FaceMeshVVADExample.py) - Detect voice activation by landmark sequence classification.

There are even more examples where visiongraph is currently in use:

- [Spout/Syphon RGB-D Example](https://github.com/cansik/spout-rgbd-example) - Share RGB-D images over spout or syphon.
- [WebRTC Input](https://github.com/cansik/visiongraph-webrtc) - WebRTC input example for visiongraph

## Development
To develop on visiongraph it is recommended to clone this repository and install the dependencies like this. First install the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager.

```bash
# in the visiongraph directory install all dependencies
uv sync --all-extras --dev --group docs
```

### Build
To build a new wheel package of visiongraph run the following command in the root directory. Please find the wheel and source distribution in `./dist`.

```bash
uv run python setup.py generate_init
uv build
```

### Docs

To generate the documentation, use the following commands.

```bash
# create documentation into "./docs
uv run python setup.py doc

# launch pdoc webserver
uv run python setup.py doc --launch
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
Copyright (c) 2025 Florian Bruggisser