import multiprocessing as mp
from argparse import Namespace, ArgumentParser
from typing import TypeVar, Optional

from visiongraph.GraphNode import GraphNode

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class AsyncGraphNode(GraphNode[InputType, OutputType]):
    """
    An asynchronous graph node that runs in a separate process.

    This class provides an implementation of the graph node interface
    with support for asynchronous execution.
    """

    def __init__(self, node: GraphNode[InputType, OutputType],
                 input_queue_size: int = 1, output_queue_size: int = 1,
                 daemon: bool = True):
        """
        Initializes an instance of the AsyncGraphNode class.

        :param node: The underlying graph node.
        :param input_queue_size: The maximum size of the input queue. Defaults to 1.
        :param output_queue_size: The maximum size of the output queue. Defaults to 1.
        :param daemon: Whether the process should be a daemon. Defaults to True.
        """
        self.node = node

        self.daemon = daemon

        self.input_queue = mp.Queue(maxsize=input_queue_size)
        self.output_queue = mp.Queue(maxsize=output_queue_size)

        self._loop_executor: Optional[mp.Process] = None
        self._running = False

    def setup(self):
        """
        Starts the process and begins the loop.

        This method should be called before any other methods on the
        instance are invoked.
        """
        self._running = True

        self._loop_executor = mp.Process(target=self._graph_loop, daemon=self.daemon)
        self._loop_executor.start()

    def _graph_loop(self):
        """
        The main loop of the process.

        This method is responsible for setting up the underlying graph node,
        processing input data, and sending output to the output queue.
        """
        self.node.setup()

        while self._running:
            try:
                data = self.input_queue.get(timeout=0.1)
            except TimeoutError:
                continue

            result = self.node.process(data)

            try:
                self.output_queue.put(result, timeout=5)
            except TimeoutError:
                continue

        self.node.release()

    def process(self, data: InputType) -> OutputType:
        """
        Processes input data and sends output to the output queue.

        :param data: The input data to be processed.

        :return: The output of the processing operation.
        """
        self.input_queue.put(data)
        return self.output_queue.get()

    def release(self):
        """
        Stops the process and waits for it to terminate.

        This method should be called when the instance is no longer needed.
        """
        self._running = False
        self._loop_executor.join(60 * 1)

    def configure(self, args: Namespace):
        """
        Configures the underlying graph node based on the provided arguments.

        :param args: The namespace containing the configuration options.
        """
        super().configure(args)
        self.node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters to the parser.

        This method should be overridden by subclasses to provide custom
        command-line options.
        """
        super().add_params(parser)
