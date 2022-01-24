import argparse
import visiongraph as vg


def main():
    pipeline = vg.create_graph(name="Instance Segmentation", handle_signals=True) \
        .apply(ssd=vg.sequence(vg.MaskRCNNEstimator.create()),
               image=vg.passthrough()) \
        .then(vg.ResultAnnotator(image="image"), vg.ImagePreview()) \
        .build()
    pipeline.configure(args)

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisionGraph Example", description="Example Pipeline")
    vg.VisionGraph.add_params(parser)
    vg.ObjectDetectionTracker.add_params(parser)
    args = parser.parse_args()

    main()
