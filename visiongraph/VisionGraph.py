from argparse import ArgumentParser, Namespace
from typing import Optional

from visiongraph.BaseGraph import BaseGraph
from visiongraph.GraphNode import GraphNode
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.result.BaseResult import BaseResult
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class VisionGraph(BaseGraph):

    def __init__(self, input: Optional[BaseInput] = None,
                 name: str = "VisionPipeline", skip_none_frame: bool = True,
                 multi_threaded: bool = False, daemon: bool = False, handle_signals: bool = False,
                 new_process: bool = False, *nodes: GraphNode):
        super().__init__(multi_threaded, daemon, handle_signals, new_process)

        self.input: Optional[BaseInput] = input
        self.fps = FPSTracer()

        # add nodes
        if self.input is not None:
            self.nodes.append(self.input)
        self.nodes = self.nodes + list(nodes)

        self.name = name
        self.skip_none_frame = skip_none_frame

    def _init(self):
        if self.input not in self.nodes:
            self.nodes.insert(0, self.input)

        super()._init()

    def _process(self):
        result: Optional[BaseResult] = self._inference()
        self.fps.update()

    def _inference(self) -> Optional[BaseResult]:
        result = None
        for i, node in enumerate(self.nodes):
            result = node.process(result)

            if not self.skip_none_frame and i == 0 and result is None:
                self._open = False
                return None

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
