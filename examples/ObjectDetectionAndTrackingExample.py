import argparse
from argparse import ArgumentParser
from collections import defaultdict, deque

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.DEIMv2Detector import DEIMv2Detector, DEIMv2Config
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.NMSOptions import NMSOptions
from visiongraph.tracker.FlateTracker import FlateTracker
from visiongraph.tracker.ObjectAssignmentSolver import ObjectAssignmentSolver
from visiongraph.util.DrawingUtils import COLOR_SEQUENCE
from visiongraph.util.LoggingUtils import add_logging_parameter, setup_logging


class ObjectDetectionAndTrackingExample(BaseGraph):
    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input

        self.network = DEIMv2Detector.create(DEIMv2Config.DEIMv2_HgNetv2_Pico_COCO)
        self.network.nms_options = NMSOptions()

        self.tracker = FlateTracker(class_aware=True, cost_function=ObjectAssignmentSolver.iou_cost_function)
        self.tracker.include_stale = False

        self.add_nodes(self.input, self.network, self.tracker)

        self.track_history = defaultdict(lambda: deque(maxlen=30))

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        h, w = frame.shape[:2]

        results = self.network.process(frame)
        results = self.tracker.process(results)

        active_track_ids = set()
        for result in results:
            result.annotate(frame)

            if result.tracking_id >= 0:
                active_track_ids.add(result.tracking_id)
                center = result.bounding_box.center
                # center is relative, so we scale it to image dimensions
                center_x = int(center.x * w)
                center_y = int(center.y * h)
                self.track_history[result.tracking_id].append((center_x, center_y))

        # Clean up old tracks that are no longer active
        inactive_track_ids = set(self.track_history.keys()) - active_track_ids
        for track_id in inactive_track_ids:
            del self.track_history[track_id]

        # draw the tracks
        for track_id, history in self.track_history.items():
            if len(history) > 1:
                color = COLOR_SEQUENCE[track_id % len(COLOR_SEQUENCE)]

                # convert to list of points
                points = np.array(history, dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(p: ArgumentParser):
        pass


def main():
    pipeline = ObjectDetectionAndTrackingExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Object Detection And Tracking Example", description="Example Object Detection Pipeline"
    )
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)
    ObjectDetectionAndTrackingExample.add_params(parser)

    args = parser.parse_args()

    setup_logging(args.loglevel)

    main()
