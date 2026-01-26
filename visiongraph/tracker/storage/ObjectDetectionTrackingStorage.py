from collections import deque
from typing import TypeVar, Generic, Deque, Tuple

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.storage.SimpleTrackingStorage import SimpleTrackingStorage, SimpleTrackable

T = TypeVar("T", bound=ObjectDetectionResult)


class ObjectDetectionTrack(SimpleTrackable[T], ObjectDetectionResult, Generic[T]):
    """
    A trackable object that extends object detection results with tracking capabilities.

    Maintains state such as tracking history, age, and hit count for a detected object.
    """

    def __init__(self, detection: T, maxlen: int = 30):
        """
        Initializes an ObjectDetectionTrack with an initial detection.

        :param detection: The initial object detection result.
        :param maxlen: The maximum number of past positions to store in history.
        """
        super().__init__(detection.class_id, detection.class_name, detection.score, detection.bounding_box)
        self.tracking_id = detection.tracking_id
        self.staleness = detection.staleness
        self.age = 0
        self.hits = 1
        self.history: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        self.history.append((detection.bounding_box.center.x, detection.bounding_box.center.y))

    def update_track(self, detection: T):
        """
        Updates the track with a new detection.

        :param detection: The new detection to update this track with.
        """
        self.bounding_box = detection.bounding_box
        self.score = detection.score
        self.staleness = detection.staleness
        self.age += 1
        self.hits += 1
        self.history.append((detection.bounding_box.center.x, detection.bounding_box.center.y))


class ObjectDetectionTrackingStorage(SimpleTrackingStorage[T, ObjectDetectionTrack[T]]):
    """
    A tracking storage implementation for object detection results.

    Creates and manages ObjectDetectionTrack instances to track object states over time.
    """

    def __init__(self, maxlen: int = 30):
        """
        Initializes the ObjectDetectionTrackingStorage with a maximum history length.

        :param maxlen: The maximum number of past positions to store in each track.
        """
        super().__init__()
        self.maxlen = maxlen

    def _on_create(self, detection: T) -> ObjectDetectionTrack[T]:
        """
        Creates a new ObjectDetectionTrack from a detection.

        :param detection: The detection to base the new track on.

        :return: A new ObjectDetectionTrack instance.
        """
        return ObjectDetectionTrack(detection, maxlen=self.maxlen)
