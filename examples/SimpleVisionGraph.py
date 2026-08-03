from visiongraph import vg


def main():
    graph = (
        vg.create_graph(name="Object Detection", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .apply(ssd=vg.sequence(vg.YOLOv5Detector.create(), vg.CentroidTracker()), image=vg.passthrough())
        .then(vg.ResultAnnotator(), vg.ImagePreview())
        .build()
    )
    graph.open()


if __name__ == "__main__":
    main()
