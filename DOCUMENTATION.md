# Documentation
This documentation is intended to provide an overview of the framework. A full documentation will be available later.

### Import Visiongraph
There are two ways on how to import visiongraph related objects and classes. The classical way is to use the direct import like this:

```python
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine

engine = OpenVinoEngine(...)
```

However, due to the amount of packages and package depth in visiongraph, it is recommended to use the `vg` package:

```python
from visiongraph import vg

engine = vg.OpenVinoEngine(...)
```

#### Optional Imports

`vg` allows for direct access of all members of visiongraph and even handles optional imports. If an import is not available, a stub-object is returned which throws an error on accessing its attributes. The reason behind this is, that it is possible to work with objects types, which would not be accessable on certain systems (like MacOS):

```python
from visiongraph import vg

device = ...

if isinstance(device, vg.AzureKinectInput):
    # would always be "False" on MacOS
    print("This is a Kinect")
```

### Graph
The core component of visiongraph is the [BaseGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/BaseGraph.py) class. It contains and handles all the nodes of the graph. A BaseGraph can run on the same thread as called or a new thread or process. The nodes in the graph are just a list, the graph itself is created by nesting nodes into each other.

#### Graph Node
A [GraphNode](https://github.com/cansik/visiongraph/blob/main/visiongraph/GraphNode.py) is a single step in the graph. It has a input and output type and processes the data within the `process()` method.

#### Graph Builder
The graph builder helps to create new graphs on a single line in python. It creates a [VisionGraph](https://github.com/cansik/visiongraph/blob/main/visiongraph/VisionGraph.py) object which is a child of the BaseGraph. The following code snippet is an example of the graph builder which creates a smooth pose estimation graph.

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

### Logging
To enable logging for visiongraph imports please set the following environment variable:

```bash
# zsh / bash
export VISIONGRAPH_LOGLEVEL=INFO

# cmd
set VISIONGRAPH_LOGLEVEL=INFO

# powershell
$env:VISIONGRAPH_LOGLEVEL="INFO"
```

### Extras
It is possible to install extra module to visiongraph by specifying them when installing visiongraph. Here is a list of currently supported extras:

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
