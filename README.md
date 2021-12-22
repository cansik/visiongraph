# Vision Graph
Vision Graph is a simple computer vision pipeline.

## Installation
To install visiongraph with all dependencies call pip like this:

```bash
pip install "visiongraph[all]"
```

It is also possible to only install certain packages depending on your needs:

```bash
pip install "visiongraph[realsense, openvino, mediapipe, onnx, media]"
```

### Development

To develop on visiograph itself it is recommended to clone this repository and install the dependencies like this:

```bash
# in the visiongraph directory
pip install -e ".[all]"
```

### Build

To build a new wheel package of visiongraph run the following command in the root directory.

```bash
python setup.py bdist_wheel
```

## Example

## Content

### Pipeline

### Input

Supported are video, webcam, RealSense and Azure Kinect input types. Azure Kinect may need a [special install](https://github.com/etiennedub/pyk4a#windows) on Windows and the [Azure-Kinect-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK)

### Estimator

### Tracker

## About
Copyright (c) 2021 Florian Bruggisser