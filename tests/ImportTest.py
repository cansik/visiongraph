import visiongraph as vg

if __name__ == "__main__":
    graph = vg.create_graph(name="Object Detection", input_node=vg.VideoCaptureInput(1), handle_signals=True) \
        .apply(ssd=vg.sequence(vg.SSDDetector.create(), vg.CentroidTracker()), image=vg.passthrough()) \
        .then(vg.ResultAnnotator(image="image"), vg.ImagePreview()) \
        .build()
    print("import worked")
