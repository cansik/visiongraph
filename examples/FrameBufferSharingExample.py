from visiongraph import vg

if __name__ == "__main__":
    graph = (vg.create_graph(name="Smooth Pose Estimation",
                             input_node=vg.VideoCaptureInput(0),
                             handle_signals=True)
             .apply(ssd=vg.sequence(vg.OpenPoseEstimator.create()), image=vg.passthrough())
             .then(vg.ResultAnnotator(), vg.ImagePreview())
             ).then(vg.extract("image"), vg.FrameBufferSharingServer.create("visiongraph"))
    graph.open()
