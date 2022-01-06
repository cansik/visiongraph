import argparse
from typing import Dict, List

from visiongraph.VisionGraph import VisionGraph
from visiongraph.estimator.ChainEstimator import ChainEstimator
from visiongraph.estimator.spatial.face.AdasFaceDetector import AdasFaceDetector
from visiongraph.result.BaseResult import BaseResult
from visiongraph.result.spatial.face.FaceDetectionResult import FaceDetectionResult
from visiongraph.tracker.ObjectDetectionTracker import ObjectDetectionTracker


def on_results_ready(results: Dict[str, BaseResult]):
    faces: List[FaceDetectionResult] = results["facenet"]
    print(f"Faces detected: {len(faces)}")


def main():
    pipeline = VisionGraph(name="Face Detection", multi_threaded=False,
                           facenet=ChainEstimator(AdasFaceDetector.create(), ObjectDetectionTracker()))
    pipeline.configure(args)
    pipeline.on_results_ready = on_results_ready

    pipeline.open()
    pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisionGraph Example", description="Example Pipeline")
    VisionGraph.add_params(parser)
    args = parser.parse_args()

    main()
