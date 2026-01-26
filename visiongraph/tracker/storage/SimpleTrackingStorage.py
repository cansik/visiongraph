from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import List, TypeVar, Generic

from visiongraph.GraphNode import GraphNode
from visiongraph.result.ResultList import ResultList
from visiongraph.tracker.storage.TrackingStorage import T, TrackingStorage


class SimpleTrackable(Generic[T], ABC):
    """
    Abstract base class for trackable objects.
    """

    @abstractmethod
    def update_track(self, detection: T):
        """
        Update the current trackable object with data from another trackable object.

        :param detection: The trackable object to update from.
        """
        pass


K = TypeVar("K", bound=SimpleTrackable)


class SimpleTrackingStorage(TrackingStorage[T, K], GraphNode[List[T], ResultList[K]], ABC):
    """
    A simple tracking storage implementation that integrates tracking and graph processing.

    Tracks objects by associating detections with existing tracks using a customizable update logic.
    """

    def __init__(self):
        """
        Initializes the SimpleTrackingStorage with default creation and update behaviors.
        """
        super().__init__(self._on_create, self._on_update)

    @abstractmethod
    def _on_create(self, detection: T) -> K:
        """
        Handles the creation of a new track.

        :param detection: The detected object to be stored as a new track.

        :return: The newly created track.
        """
        pass

    def _on_update(self, detection: T, track: K) -> K:
        """
        Handles the update of an existing track with a new detection.

        :param detection: The detected object.
        :param track: The existing track to be updated.

        :return: The updated track after incorporating the detected object's data.
        """
        track.update_track(detection)
        return track

    def setup(self) -> None:
        """
        Sets up the tracking storage. No additional setup is required for this implementation.
        """
        pass

    def process(self, detections: List[T]) -> ResultList[K]:
        """
        Processes a list of detections, updating tracks accordingly.

        :param detections: A list of detected objects.

        :return: A result list containing the current state of all tracks.
        """
        self.update(detections)
        return ResultList(list(self.tracks.values()))

    def release(self) -> None:
        """
        Releases resources held by the tracking storage. No additional resources to release in this implementation.
        """
        pass

    def configure(self, args: Namespace):
        """
        Configures the tracking storage using command-line arguments.

        :param args: Namespace containing configuration arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command-line parameters for configuring the tracking storage.

        :param parser: The argument parser to configure.
        """
        pass
