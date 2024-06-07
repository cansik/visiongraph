import argparse

import visiongraph as vg
from visiongraph.data.labels.COCO import COCO_80_LABELS


def main():
    segmentation_model = vg.YOLOv8SegmentationEstimator.create()
    segmentation_model.allowed_classes = {COCO_80_LABELS.index("person")}

    pipeline = vg.create_graph(name="Instance Segmentation", handle_signals=True) \
        .apply(ssd=vg.sequence(segmentation_model),
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
