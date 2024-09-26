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
    """
    A class representing a graph for machine learning pipelines.
    """

    def __init__(self, input: Optional[BaseInput] = None,
                 name: str = "VisionPipeline", skip_none_frame: bool = True,
                 multi_threaded: bool = False, daemon: bool = False, handle_signals: bool = False,
                 new_process: bool = False, *nodes: GraphNode):
        """
        Initializes the VisionGraph object.

        Args:
            input (BaseInput, optional): The input to the pipeline. Defaults to None.
            name (str): The name of the graph.
            skip_none_frame (bool): Whether to skip frames with no output. Defaults to True.
            multi_threaded (bool): Whether to run in multiple threads. Defaults to False.
            daemon (bool): Whether to run as a daemon process. Defaults to False.
            handle_signals (bool): Whether to handle signals. Defaults to False.
            new_process (bool): Whether to create a new process. Defaults to False.
            *nodes: The nodes to add to the graph.
        """
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
        """
        Initializes the graph by adding the input node to the beginning.
        """
        if self.input not in self.nodes:
            self.nodes.insert(0, self.input)

        super()._init()

    def _process(self):
        """
        Processes the graph by calling each node's process method.

        Returns:
            Optional[BaseResult]: The result of the graph processing.
        """
        result: Optional[BaseResult] = self._inference()
        self.fps.update()

    def _inference(self) -> Optional[BaseResult]:
        """
        Performs inference on the graph by calling each node's process method.

        Returns:
            Optional[BaseResult]: The result of the graph processing.
        """
        result = None
        for i, node in enumerate(self.nodes):
            result = node.process(result)

            if not self.skip_none_frame and i == 0 and result is None:
                self._open = False
                return None

        return result

    def configure(self, args: Namespace):
        """
        Configures the graph based on the provided arguments.

        Args:
            args (Namespace): The parsed command-line arguments.
        """
        super().configure(args)

        if self.input is None:
            self.input = args.input()
            self.input.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser for logging and input provider.

        Args:
            parser (ArgumentParser): The parser to add parameters to.
        """
        add_logging_parameter(parser)
        input_group = parser.add_argument_group("input provider")
        add_input_step_choices(input_group)
