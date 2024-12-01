from collections import OrderedDict
import numpy as np
# todo: get rid of scipy
from scipy.spatial import distance
from visiongraph.external.motrackers.utils.misc import get_centroid
from visiongraph.external.motrackers.Track import Track


class Tracker:
    """
    Greedy Tracker with tracking based on ``centroid`` location of the bounding box of the object.
    This tracker is also referred as ``CentroidTracker`` in this repository.

    :param max_lost: Maximum number of consecutive frames object was not detected.
    :param tracker_output_format: Output format of the tracker.
    """

    def __init__(self, max_lost=5, tracker_output_format='mot_challenge'):
        self.next_track_id = 0
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.tracker_output_format = tracker_output_format

    def _add_track(self, frame_id, bbox, detection_confidence, class_id, **kwargs):
        """
        Add a newly detected object to the queue.

        :param frame_id: Camera frame id.
        :param bbox: Bounding box pixel coordinates as (xmin, ymin, xmax, ymax) of the track.
        :param detection_confidence: Detection confidence of the object (probability).
        :param class_id: Class label id.
        :param kwargs: Additional key word arguments.
        """

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        self.next_track_id += 1

    def _remove_track(self, track_id):
        """
        Remove tracker data after object is lost.

        :param track_id: track_id of the track lost while tracking.
        """

        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0.,
                      reference=-1, **kwargs):
        """
        Update track state.

        :param track_id: ID of the track.
        :param frame_id: Frame count.
        :param bbox: Bounding box coordinates as `(xmin, ymin, width, height)`.
        :param detection_confidence: Detection confidence (a.k.a. detection probability).
        :param class_id: ID of the class (aka label) of the object being tracked.
        :param lost: Number of frames the object was lost while tracking.
        :param iou_score: Intersection over union.
        :param kwargs: Additional keyword arguments.
        """

        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score,
            reference=reference, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        """
        Output the information of tracks.

        :param tracks: Tracks dictionary with (key, value) as (track_id, corresponding `Track` objects).

        :return: List of tracks being currently tracked by the tracker.
        """

        outputs = []
        for trackid, track in tracks.items():
            if not track.lost:
                outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores, references):
        """
        Preprocess the input data.

        :param bboxes: Array of bounding boxes with each bbox as a tuple containing `(xmin, ymin, width, height)`.
        :param class_ids: Array of Class ID or label ID.
        :param detection_scores: Array of detection scores (a.k.a. detection probabilities).

        """

        new_bboxes = np.array(bboxes, dtype='float')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)
        new_references = np.array(references, dtype='int')

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores, new_references))
        return new_detections

    def update(self, bboxes, detection_scores, class_ids, references):
        """
        Update the tracker based on the new bounding boxes.

        :param bboxes: List of bounding boxes detected in the current frame. Each element of the list represent
        :param detection_scores: List of detection scores (probability) of each detected object.
        :param class_ids: List of class_ids (int) corresponding to labels of the detected object. Default is `None`.

        :return: List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores, references)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(np.asarray(bboxes))

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence, reference = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id,
                                       reference=reference)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence, reference) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id, reference=reference)

        outputs = self._get_tracks(self.tracks)
        return outputs
