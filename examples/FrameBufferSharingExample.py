from visiongraph import vg


def main():
    graph = (
        vg.create_graph(name="Frame Buffer Sharing", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .apply(poses=vg.sequence(vg.OpenPoseEstimator.create()), image=vg.passthrough())
        .then(vg.ResultAnnotator(), vg.ImagePreview())
        .then(vg.extract("image"), vg.FrameBufferSharingServer.create("visiongraph"))
        .build()
    )
    graph.open()


if __name__ == "__main__":
    main()
