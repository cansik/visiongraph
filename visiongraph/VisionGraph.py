from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, Callable

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.VisionEstimator import VisionEstimator
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class VisionGraph(BaseGraph):

    def __init__(self, input: Optional[BaseInput] = None,
                 name: str = "VisionPipeline", annotate: bool = True, display: bool = True,
                 multi_threaded: bool = False, deamon: bool = False, handle_signals: bool = False,
                 **estimators: VisionEstimator):
        super().__init__(multi_threaded, deamon, handle_signals)

        self.input: Optional[BaseInput] = input
        self.estimators: Dict[str, VisionEstimator] = dict() if estimators is None else estimators
        self.fps = FPSTracer()

        self.name = name
        self.display = display
        self.annotate = annotate

        # events
        self.on_frame_ready: Optional[Callable[[int, np.ndarray], None]] = None
        self.on_results_ready: Optional[Callable[[Dict[str, BaseResult]], None]] = None

    def _init(self):
        # add nodes
        self.add_nodes(self.input, *self.estimators.values())

        super()._init()

    def _process(self):
        # read frame
        ts, frame = self.input.read()

        if frame is None:
            return

        if self.on_frame_ready is not None:
            self.on_frame_ready(ts, frame)

        results: Dict[str, BaseResult] = self._inference(frame)
        self.fps.update()

        if self.on_results_ready is not None:
            self.on_results_ready(results)

        # annotate
        if self.annotate:
            for result in results.values():
                if isinstance(result, list):
                    for r in result:
                        r.annotate(frame)
                    continue

                result.annotate(frame)

        # analyse
        if self.display:
            cv2.imshow(self.name, frame)
            if cv2.waitKey(15) & 0xFF == 27:
                self.close()

    def configure(self, args: Namespace):
        super().configure(args)

        if self.input is None:
            self.input = args.input()
            self.input.configure(args)

    def _inference(self, frame: np.ndarray) -> Dict[str, BaseResult]:
        results: Dict[str, BaseResult] = dict()
        for name, estimator in self.estimators.items():
            results[name] = estimator.process(frame)
        return results

    @staticmethod
    def add_params(parser: ArgumentParser):
        add_logging_parameter(parser)
        input_group = parser.add_argument_group("input provider")
        add_input_step_choices(input_group)
