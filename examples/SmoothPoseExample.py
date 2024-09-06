# smooth pose example
import sys

from visiongraph import vg

if __name__ == "__main__":
    graph = (vg.create_graph(name="Smooth Pose Estimation",
                             input_node=vg.VideoCaptureInput(sys.argv[1]),
                             handle_signals=True)
             .apply(ssd=vg.sequence(vg.OpenPoseEstimator.create(), vg.MotpyTracker(), vg.LandmarkSmoothFilter()),
                    image=vg.passthrough())
             .then(vg.ResultAnnotator(image="image"), vg.ImagePreview())
             )
    graph.open()
