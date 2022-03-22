from argparse import Namespace, ArgumentParser
from typing import TypeVar, Optional

from visiongraph.GraphNode import GraphNode
import multiprocessing as mp

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class AsyncGraphNode(GraphNode[InputType, OutputType]):
    def __init__(self, node: GraphNode[InputType, OutputType],
                 input_queue_size: int = 1, output_queue_size: int = 1,
                 daemon: bool = True):
        self.node = node

        self.daemon = daemon

        self.input_queue = mp.Queue(maxsize=input_queue_size)
        self.output_queue = mp.Queue(maxsize=output_queue_size)

        self._loop_executor: Optional[mp.Process] = None
        self._running = False

    def setup(self):
        self._running = True

        self._loop_executor = mp.Process(target=self._graph_loop, daemon=self.daemon)
        self._loop_executor.start()

    def _graph_loop(self):
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
        self.input_queue.put(data)
        return self.output_queue.get()

    def release(self):
        self._running = False
        self._loop_executor.join(60 * 1)

    def configure(self, args: Namespace):
        super().configure(args)
        self.node.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        super().add_params(parser)
