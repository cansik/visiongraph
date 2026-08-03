import argparse

from visiongraph import vg


def main():
    parser = argparse.ArgumentParser("Smooth Pose Example", description="Estimate and smooth human poses.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    graph = (
        vg.create_graph(name="Smooth Pose Estimation", input_node=args.input(), handle_signals=True)
        .apply(
            poses=vg.sequence(vg.OpenPoseEstimator.create(), vg.MotpyTracker(), vg.LandmarkSmoothFilter()),
            image=vg.passthrough(),
        )
        .then(vg.ResultAnnotator(image="image"), vg.ImagePreview())
        .build()
    )
    graph.configure(args)
    graph.open()


if __name__ == "__main__":
    main()
