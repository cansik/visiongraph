import argparse
import visiongraph as vg


def main():
    pipeline = vg.create_graph(name="Instance Segmentation", handle_signals=True) \
        .apply(ssd=vg.sequence(vg.YOLOv8SegmentationEstimator.create()),
               image=vg.passthrough()) \
        .then(vg.ResultAnnotator(), vg.ImagePreview()) \
        .build()
    pipeline.configure(args)

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisionGraph Example", description="Example Pipeline")
    vg.VisionGraph.add_params(parser)
    vg.CentroidTracker.add_params(parser)
    args = parser.parse_args()

    main()
