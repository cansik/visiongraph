import argparse
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import tqdm
from visiongraph import vg
from visiongraph.input import add_input_step_choices
from visiongraph.util.DrawingUtils import COLOR_SEQUENCE


class BaseReadmeGraph(vg.BaseGraph, ABC):
    def __init__(self, input: vg.BaseInput):
        super().__init__()
        self.input = input

        self.tracker = vg.FlateTracker(class_aware=True, cost_function=vg.ObjectAssignmentSolver.iou_cost_function)
        self.tracker.include_stale = False

        self.storage = vg.ObjectDetectionTrackingStorage(maxlen=30)

        self.recorder = vg.CV2VideoRecorder(None, None)

        self.add_nodes(self.input, self.tracker, self.storage, self.recorder)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            self.close()
            return

        results = self.detect(frame)
        results = self.tracker.process(results)
        tracks = self.storage.process(results)

        self.annotate(frame, results, tracks)
        self.recorder.add_image(frame)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    @abstractmethod
    def detect(self, frame: np.ndarray) -> vg.ResultList[vg.ObjectDetectionResult]:
        pass

    @abstractmethod
    def annotate(
        self,
        frame: np.ndarray,
        results: vg.ResultList[vg.ObjectDetectionResult],
        tracks: vg.ResultList[vg.ObjectDetectionTrack],
    ):
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def add_params(p: ArgumentParser):
        pass


class PoseGraph(BaseReadmeGraph):
    def __init__(self, input: vg.BaseInput):
        super().__init__(input)
        self.network = vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_S_COCO_1280)
        self.network.min_score = 0.1
        self.network = vg.UltralyticsPoseEstimator.create(vg.UltralyticsPoseConfig.YOLOv11_S_640)

        self.add_nodes(self.network)

    def detect(self, frame: np.ndarray) -> vg.ResultList[vg.ObjectDetectionResult]:
        return self.network.process(frame)

    def annotate(
        self,
        frame: np.ndarray,
        results: vg.ResultList[vg.ObjectDetectionResult],
        tracks: vg.ResultList[vg.ObjectDetectionTrack],
    ):
        results.annotate(frame, show_bounding_box=False)


class SegmentationGraph(BaseReadmeGraph):
    def __init__(self, input: vg.BaseInput):
        super().__init__(input)

        self.network = vg.YOLOv8SegmentationEstimator.create(vg.YOLOv8SegmentationConfig.YOLOv8_SEG_L)
        self.add_nodes(self.network)

    def detect(self, frame: np.ndarray) -> vg.ResultList[vg.ObjectDetectionResult]:
        return self.network.process(frame)

    def annotate(
        self,
        frame: np.ndarray,
        results: vg.ResultList[vg.ObjectDetectionResult],
        tracks: vg.ResultList[vg.ObjectDetectionTrack],
    ):
        results.annotate(frame, show_bounding_box=False)


class ObjectDetectionGraph(BaseReadmeGraph):
    def __init__(self, input: vg.BaseInput):
        super().__init__(input)

        # model = vg.DEIMv2Detector.create(vg.DEIMv2Config.DEIMv2_Dino3_S_COCO)
        model = vg.YOLOv8Detector.create(vg.YOLOv8Config.YOLOv8_L)
        model.min_score = 0.5

        self.network = model
        # self.network = vg.SlidingWindowEstimator(model, 320, (640, 480), 0.8)

        self.add_nodes(model)

    def detect(self, frame: np.ndarray) -> vg.ResultList[vg.ObjectDetectionResult]:
        return self.network.process(frame)

    def annotate(
        self,
        frame: np.ndarray,
        results: vg.ResultList[vg.ObjectDetectionResult],
        tracks: vg.ResultList[vg.ObjectDetectionTrack],
    ):
        h, w = frame.shape[:2]
        for track in tracks:
            track.annotate(frame)

            if len(track.history) > 1:
                color = COLOR_SEQUENCE[track.tracking_id % len(COLOR_SEQUENCE)]

                # convert to list of points
                points = []
                for x, y in track.history:
                    points.append((int(x * w), int(y * h)))

                points = np.array(points, dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


class FaceDetectionGraph(BaseReadmeGraph):
    def __init__(self, input: vg.BaseInput):
        super().__init__(input)

        self.network = vg.SpatialCascadeEstimator(
            vg.AdasFaceDetector.create(vg.AdasFaceConfig.MobileNet_672x384_FP32),
            landmarks=vg.RegressionLandmarkEstimator(),
            head_pose=vg.AdasHeadPoseEstimator(),
            emotion2=vg.FERPlusEmotionClassifier(),
        )
        self.add_nodes(self.network)

    def detect(self, frame: np.ndarray) -> vg.ResultList[vg.ObjectDetectionResult]:
        return self.network.process(frame)

    def annotate(
        self,
        frame: np.ndarray,
        results: vg.ResultList[vg.ObjectDetectionResult],
        tracks: vg.ResultList[vg.ObjectDetectionTrack],
    ):
        results.annotate(frame)


def main():
    # argument parsing
    parser = argparse.ArgumentParser(description="Readme Video Generation")
    vg.add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)
    parser.add_argument("-o", "--output", default="media/readme", help="Output path of the videos.")
    args = parser.parse_args()

    vg.setup_logging(args.loglevel)

    graphs = [ObjectDetectionGraph, PoseGraph, FaceDetectionGraph, SegmentationGraph]

    output_path_root = Path(args.output)

    for graph_type in tqdm.tqdm(graphs, desc="processing"):
        # setup input
        input_node: vg.BaseInput = args.input()

        graph = graph_type(input_node)

        graph.configure(args)
        graph_name = graph.name

        output_path = output_path_root / f"{graph_name.lower()}.mp4"

        if isinstance(input_node, vg.VideoCaptureInput):
            input_node.loop = False
            video_stem = Path(input_node.channel).stem
            output_path = output_path_root / video_stem / f"{video_stem}-{output_path.name}"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        graph.recorder.output_path = str(output_path)
        graph.open()

    print("done!")


if __name__ == "__main__":
    main()
