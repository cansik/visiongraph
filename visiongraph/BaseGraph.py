import logging
import signal
from abc import ABC, abstractmethod
from argparse import Namespace
from multiprocessing import Process
from threading import Thread
from typing import List, Callable, Optional, Union

from visiongraph.model.parameter.ArgumentConfigurable import ArgumentConfigurable
from visiongraph.GraphNode import GraphNode


class BaseGraph(ArgumentConfigurable, ABC):
    def __init__(self, multi_threaded: bool = False, daemon: bool = False,
                 handle_signals: bool = False, new_process: bool = False):
        self._open = False
        self.multi_threaded = multi_threaded
        self.new_process = new_process
        self.daemon = daemon
        self._loop_executor: Optional[Union[Thread, Process]] = None
        self.nodes: List[GraphNode] = []

        self.on_exception: Optional[Callable[[BaseGraph, Exception], None]] = None

        if handle_signals:
            signal.signal(signal.SIGINT, self._signal_handler)

    def add_nodes(self, *nodes: GraphNode):
        self.nodes += nodes

    def open(self):
        if self._open:
            logging.warning("is already running")
            return

        logging.info("open pipeline...")
        self._open = True

        if self.multi_threaded:
            if self.new_process:
                self._loop_executor = Process(target=self._loop, daemon=self.daemon)
                self._loop_executor.start()
            else:
                self._loop_executor = Thread(target=self._loop, daemon=self.daemon)
                self._loop_executor.start()
        else:
            self._loop()

    def close(self, wait_time: int = 60 * 1000):
        if not self._open:
            logging.warning("is not running")
            return

        logging.info(f"closing {self.__class__.__name__}...")
        self._open = False

        if self.multi_threaded:
            self._loop_executor.join(wait_time)

        logging.info("has been closed")

    def _loop(self):
        try:
            self._init()
            logging.info("is setup and running")

            while self._open:
                self._process()

            self._release()
        except Exception as ex:
            if self.on_exception is None:
                raise ex
            self.on_exception(self, ex)

    def _init(self):
        """Runs before pipeline loop."""
        for node in self.nodes:
            node.setup()

    @abstractmethod
    def _process(self):
        """Runs inside pipeline loop."""
        pass

    def _release(self):
        """Runs after pipeline loop"""
        for node in self.nodes:
            node.release()

    def configure(self, args: Namespace):
        for node in self.nodes:
            node.configure(args)

    def _signal_handler(self, signal, frame):
        self.close()
