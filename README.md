# Vision Graph
Vision Graph is a simple computer vision pipeline.

## Installation
To install visiongraph with all dependencies call pip like this:

```bash
pip install "visiongraph[all]"
```

It is also possible to only install certain packages:

```bash
pip install "visiongraph[realsense, openvino, mediapipe, onnx, media]"
```

### Development

To develop it is recommended to clone this repository and install the dependencies like this:

```bash
# in the visiongraph directory
pip install -e ".[all]"
```

### Build

To build a new wheel package of visiongraph run the following command in the root directory.

```bash
python setup.py bdist_wheel
```

## About
Copyright (c) 2021 Florian Bruggisser