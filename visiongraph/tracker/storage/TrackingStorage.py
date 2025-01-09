from typing import Generic, TypeVar, List, Dict, Callable, Optional

from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult

T = TypeVar("T", bound=ObjectDetectionResult)  # The type of object detection result
K = TypeVar("K")  # The type of the tracking data


class TrackingStorage(Generic[T, K]):
    def __init__(self,
                 on_create: Callable[[T], K],
                 on_update: Callable[[T, K], K],
                 on_remove: Optional[Callable[[K], None]] = None):
        """
        :param on_create: Function to create a new track from a detected object.
        :param on_update: Function to update an existing track with a detected object.
        :param on_remove: Optional function to handle cleanup when a track is removed.
        """
        self.tracks: Dict[int, K] = {}  # Maps tracking IDs to tracking data
        self.on_create: Callable[[T], K] = on_create
        self.on_update: Callable[[T, K], K] = on_update
        self.on_remove: Optional[Callable[[K], None]] = on_remove

    def update(self, detected_objects: List[T]):
        """
        Updates the tracking storage with a list of tracked objects.
        :param detected_objects: List of detected objects with tracking IDs.
        """
        # Track IDs that are updated during this call
        updated_track_ids = set()

        for detection in detected_objects:
            if detection.tracking_id not in self.tracks:
                self.tracks[detection.tracking_id] = self.on_create(detection)
            else:
                value = self.tracks[detection.tracking_id]
                self.tracks[detection.tracking_id] = self.on_update(detection, value)

            updated_track_ids.add(detection.tracking_id)

        # Remove tracks that were not updated in this call
        self._remove_unupdated_tracks(updated_track_ids)

    def _remove_unupdated_tracks(self, updated_track_ids: set[int]):
        """
        Removes tracks that were not updated in the current update cycle.
        :param updated_track_ids: Set of track IDs that were updated.
        """
        removed_tracks = [track for tid, track in self.tracks.items() if tid not in updated_track_ids]

        if self.on_remove:
            for track in removed_tracks:
                self.on_remove(track)

        self.tracks = {tid: track for tid, track in self.tracks.items() if tid in updated_track_ids}
