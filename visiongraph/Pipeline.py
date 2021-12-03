import logging
import signal
from abc import ABC, abstractmethod
from argparse import Namespace
from threading import Thread
from typing import List

from visiongraph.model.parameter.ArgumentConfigurable import ArgumentConfigurable
from visiongraph.PipelineNode import PipelineNode


class Pipeline(ArgumentConfigurable, ABC):
    def __init__(self, multi_threaded: bool = True, deamon: bool = True, signal_listener: bool = True):
        self._open = False
        self.multi_threaded = multi_threaded
        self._loop_thread = Thread(target=self._loop, daemon=deamon)
        self.nodes: List[PipelineNode] = []

        if signal_listener:
            signal.signal(signal.SIGINT, self._signal_handler)

    def add_nodes(self, *nodes: PipelineNode):
        self.nodes += nodes

    def open(self):
        if self._open:
            logging.warning(f"{self.__class__.__name__} is already running")
            return

        logging.info("open pipeline...")
        self._open = True

        if self.multi_threaded:
            self._loop_thread.start()
        else:
            self._loop()

    def close(self):
        if not self._open:
            logging.warning(f"{self.__class__.__name__} is not running")
            return

        logging.info(f"closing {self.__class__.__name__}...")
        self._open = False

        if self.multi_threaded:
            self._loop_thread.join(5000)

        logging.info(f"{self.__class__.__name__} has been closed")

    def _loop(self):
        self._init()
        logging.info(f"{self.__class__.__name__} is setup and running")

        while self._open:
            self._process()

        self._release()

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
