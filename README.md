<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/f30f10ef-8058-4306-882f-6301226107be">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/0ed34695-ca0e-47ff-aebb-eb59ff851770">
  <img src="https://github.com/user-attachments/assets/0ed34695-ca0e-47ff-aebb-eb59ff851770" alt="Visiongraph Logo Bright" width="75%">
</picture>

# Visiongraph

[![Ruff](https://github.com/cansik/visiongraph/actions/workflows/code.yml/badge.svg)](https://github.com/cansik/visiongraph/actions/workflows/code.yml)
[![PyPI](https://img.shields.io/pypi/v/visiongraph)](https://pypi.org/project/visiongraph/)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/visiongraph)
[![Documentation](https://img.shields.io/badge/read-documentation-blue)](https://cansik.github.io/visiongraph/visiongraph.html#documentation)

Visiongraph is a computer vision pipeline designed to simplify prototyping of image-based algorithms with ready-to-use modules. Built on top of OpenCV, it also integrates popular frameworks such as [Intel OpenVINO](https://github.com/openvinotoolkit/openvino), [Google MediaPipe](https://github.com/google-ai-edge/mediapipe), and [DepthAI](https://pypi.org/project/depthai/). The library is optimized for real-time applications and edge performance.

Here is an example on how to start a webcam capture and display the image:

```python
from visiongraph import vg

vg.create_graph(vg.VideoCaptureInput()).then(vg.ImagePreview()).open()
```

Get started with `visiongraph` by reading the **[documentation](https://cansik.github.io/visiongraph/visiongraph.html#documentation)**.

## Installation
Visiongraph supports Python 3.10, 3.11 and 3.12. Other versions may also work, but are not officially supported. Usually this is a third-party dependency problem, and not directly connected to visiongraph.

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

This will install Visiongraph plus the official `mediapipe` package (which requires `numpy<2.0`). Make sure your environment’s NumPy version is below 2.0 when using this route.


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
- [NDI Input / Output](https://github.com/cansik/visiongraph-ndi) - Receive and share video frames over NDI
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

### Linter

```bash
ruff format && ruff check --fix
```

## Dependencies

Parts of these libraries are directly included and adapted to work with visiongraph.

* [motpy](https://github.com/wmuron/motpy) - Simple multi-object tracking library (MIT License)
* [motrackers](https://github.com/adipandas/multi-object-tracker) - Multi-object trackers in Python (MIT License)
* [OneEuroFilter-Numpy](https://github.com/HoBeom/OneEuroFilter-Numpy) - (MIT License)

Here you can find a list of the dependencies of visiongraph and their license (no guarantee of correctness):

```
depthai               MIT License
faiss-cpu             MIT License & BSD-3-Clause
filterpy              MIT License
mediapipe-numpy2      Apache License 2.0
moviepy               MIT License
numba                 BSD License
numpy                 MIT License
onnxruntime           MIT License
onnxruntime-directml  MIT License
onnxruntime-gpu       MIT License
opencv-python         Apache License 2.0
openvino              Apache License 2.0
pdoc                  MIT License
pyk4a                 MIT License
pyopengl              BSD License
pyrealsense2          Apache License 2.0
pyrealsense2-macosx   Apache License 2.0
pytest                MIT License
requests              Apache License 2.0
ruff                  MIT License
scipy                 MIT License
setuptools            MIT License
SpoutGL               BSD License
syphon-python         MIT License
tqdm                  MIT License
ty                    MIT License
vector                BSD License
vidgear               Apache License 2.0
wheel                 MIT License
```

For more information about the dependencies, have a look at the [pyproject.toml](pyproject.toml).

Please **note** that some models (such as Ultralytics YOLOv8 and YOLOv11) have specific licenses (AGPLv3). Always check the model license before using the model.

## About
Copyright (c) 2026 Florian Bruggisser  
Released under the MIT License. See [LICENSE](LICENSE) for details.