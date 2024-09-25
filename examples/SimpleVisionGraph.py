# simplest vision graph example
from visiongraph import vg

if __name__ == "__main__":
    graph = vg.create_graph(name="Object Detection", input_node=vg.VideoCaptureInput(1), handle_signals=True) \
        .apply(ssd=vg.sequence(vg.YOLOv5Detector.create(), vg.CentroidTracker()), image=vg.passthrough()) \
        .then(vg.ResultAnnotator(), vg.ImagePreview()) \
        .build()
    graph.open()
