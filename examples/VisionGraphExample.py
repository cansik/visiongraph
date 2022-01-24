import argparse
from typing import List

import visiongraph as vg


def on_results_ready(result: vg.BaseResult):
    faces: List[vg.FaceDetectionResult] = result
    print(f"Faces detected: {len(faces)}")


def main():
    pipeline = vg.create_graph(name="Face Detection", handle_signals=True) \
        .apply(ssd=vg.sequence(vg.AdasFaceDetector.create(), vg.CentroidTracker(), vg.custom(on_results_ready)),
               image=vg.passthrough()) \
        .then(vg.ResultAnnotator(image="image"), vg.ImagePreview()) \
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
