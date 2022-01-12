from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, Callable, List

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.GraphNode import GraphNode
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
                 *nodes: GraphNode):
        super().__init__(multi_threaded, deamon, handle_signals)

        self.input: Optional[BaseInput] = input
        self.fps = FPSTracer()

        # add nodes
        if self.input is not None:
            self.nodes.append(self.input)
        self.nodes = self.nodes + list(nodes)

        self.name = name
        self.display = display
        self.annotate = annotate

    def _init(self):
        if self.input not in self.nodes:
            self.nodes.insert(0, self.input)

        super()._init()

    def _process(self):
        result: BaseResult = self._inference()
        self.fps.update()

    def _inference(self) -> BaseResult:
        result = None
        for node in self.nodes:
            result = node.process(result)
        return result

    def configure(self, args: Namespace):
        super().configure(args)

        if self.input is None:
            self.input = args.input()
            self.input.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        add_logging_parameter(parser)
        input_group = parser.add_argument_group("input provider")
        add_input_step_choices(input_group)
