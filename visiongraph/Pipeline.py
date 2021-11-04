import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from threading import Thread
from typing import List

from visiongraph.model.parameter.ArgumentConfigurable import ArgumentConfigurable
from visiongraph.PipelineNode import PipelineStep


class Pipeline(ArgumentConfigurable, ABC):
    def __init__(self, deamon: bool = True):
        self._open = False
        self._loop_thread = Thread(target=self._loop, daemon=deamon)
        self.steps: List[PipelineStep] = []

    def add_steps(self, *steps: List[PipelineStep]):
        self.steps += steps

    def open(self):
        if self._open:
            logging.warning(f"{self.__class__.__name__} is already running")
            return

        logging.info("open pipeline...")
        self._open = True
        self._loop_thread.start()

    def close(self):
        if not self._open:
            logging.warning(f"{self.__class__.__name__} is not running")
            return

        logging.info(f"closing {self.__class__.__name__}...")
        self._open = False
        self._loop_thread.join(5000)
        logging.info(f"{self.__class__.__name__} has been closed")

    def _loop(self):
        self._init()
        logging.info(f"{self.__class__.__name__} is setup and running")

        while self._open:
            self._process()

        self._release()

    @abstractmethod
    def _init(self):
        """Runs before pipeline loop."""
        for step in self.steps:
            step.setup()

    @abstractmethod
    def _process(self):
        """Runs inside pipeline loop."""
        pass

    @abstractmethod
    def _release(self):
        """Runs after pipeline loop"""
        for step in self.steps:
            step.release()

    def configure(self, args: Namespace):
        for step in self.steps:
            step.configure(args)
