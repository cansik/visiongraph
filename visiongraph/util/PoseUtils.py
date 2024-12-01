"""
Based on https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR#scrollTo=QBrKOeP30RAx
"""

import copy
from typing import Optional

import numpy as np
import vector

from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util.VectorUtils import lerp_vector_3d, lerp_vector_2d, \
    vector_distance


def embed_pose(pose: PoseLandmarkResult, torso_size_multiplier: float = 2.5) -> Optional[np.ndarray]:
    """
Normalizes pose landmarks and converts to embedding.

    :param pose: PoseLandmarkResult with 3D landmarks.
    :param torso_size_multiplier: Multiplier to apply to the torso to get minimal body size.

    :return: Numpy array with pose embedding of shape (M, 3) where `M` is the number of
    """
    normalized_pose = _normalize_pose_landmarks(pose, torso_size_multiplier)

    # Get embedding.
    embedding = _get_pose_distance_embedding(normalized_pose)

    return embedding


def _normalize_pose_landmarks(pose: PoseLandmarkResult, torso_size_multiplier: float) -> PoseLandmarkResult:
    """
Normalizes landmarks translation and scale.
"""
    normalized_pose = copy.deepcopy(pose)

    # Normalize translation.
    pose_center = _get_pose_center(normalized_pose)
    normalized_pose.landmarks = normalized_pose.landmarks.subtract(pose_center.to_4D(t=0))

    # Normalize scale.
    pose_size = _get_pose_size(pose, torso_size_multiplier)
    normalized_pose.landmarks = normalized_pose.landmarks.scale(1 / pose_size)

    return normalized_pose


def _get_pose_center(pose: PoseLandmarkResult) -> vector.Vector3D:
    """
Calculates pose center as point between hips.

    :param pose: PoseLandmarkResult to compute the center from.

    :return: The 3D coordinates of the pose center calculated from the hips.
    """
    return lerp_vector_3d(pose.left_hip.to_xyz(), pose.right_hip.to_xyz(), 0.5)


def _get_pose_size(pose: PoseLandmarkResult, torso_size_multiplier: float) -> float:
    """
Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark

    :param pose: PoseLandmarkResult to compute the size from.
    :param torso_size_multiplier: Multiplier to apply to torso size.

    :return: Calculated pose size.
    """
    # Hips center (2D)
    hips = lerp_vector_2d(pose.left_hip.to_xy(), pose.right_hip.to_xy(), 0.5)

    # Shoulders center (2D)
    shoulders = lerp_vector_2d(pose.left_shoulder.to_xy(), pose.right_shoulder.to_xy(), 0.5)

    # Torso size as the minimum body size.
    torso_size = vector_distance(shoulders, hips)

    # Max dist to pose center.
    pose_center = _get_pose_center(pose)
    max_dist = np.max(abs(pose.landmarks.to_xyz().subtract(pose_center)))

    return max(torso_size * torso_size_multiplier, max_dist)


def _get_pose_distance_embedding(pose: PoseLandmarkResult) -> np.ndarray:
    """
Converts pose landmarks into 3D embedding.

    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We different types of pairs to cover
    different pose classes. Feel free to remove some or add new.

    :param pose: Normalized PoseLandmarkResult with 3D landmarks.

    :return: Numpy array with pose embedding of shape (M, 3) where `M` is the number of
    """
    embedding = np.array([
        # One joint.
        vector_distance(
            lerp_vector_2d(pose.left_hip.to_xyz(), pose.right_hip.to_xyz(), 0.5),
            lerp_vector_2d(pose.left_shoulder.to_xyz(), pose.right_shoulder.to_xyz(), 0.5)),

        vector_distance(pose.left_shoulder.to_xyz(), pose.left_elbow.to_xyz()),
        vector_distance(pose.right_shoulder.to_xyz(), pose.right_elbow.to_xyz()),

        vector_distance(pose.left_elbow.to_xyz(), pose.left_wrist.to_xyz()),
        vector_distance(pose.right_elbow.to_xyz(), pose.right_wrist.to_xyz()),

        vector_distance(pose.left_hip.to_xyz(), pose.left_knee.to_xyz()),
        vector_distance(pose.right_hip.to_xyz(), pose.right_knee.to_xyz()),

        vector_distance(pose.left_knee.to_xyz(), pose.left_ankle.to_xyz()),
        vector_distance(pose.right_knee.to_xyz(), pose.right_ankle.to_xyz()),

        # Two joints.
        vector_distance(pose.left_shoulder.to_xyz(), pose.left_wrist.to_xyz()),
        vector_distance(pose.right_shoulder.to_xyz(), pose.right_wrist.to_xyz()),

        vector_distance(pose.left_hip.to_xyz(), pose.left_ankle.to_xyz()),
        vector_distance(pose.right_hip.to_xyz(), pose.right_ankle.to_xyz()),

        # Four joints.
        vector_distance(pose.left_hip.to_xyz(), pose.left_wrist.to_xyz()),
        vector_distance(pose.right_hip.to_xyz(), pose.right_wrist.to_xyz()),

        # Five joints.
        vector_distance(pose.left_shoulder.to_xyz(), pose.left_ankle.to_xyz()),
        vector_distance(pose.right_shoulder.to_xyz(), pose.right_ankle.to_xyz()),

        vector_distance(pose.left_hip.to_xyz(), pose.left_wrist.to_xyz()),
        vector_distance(pose.right_hip.to_xyz(), pose.right_wrist.to_xyz()),

        # Cross body.
        vector_distance(pose.left_elbow.to_xyz(), pose.right_elbow.to_xyz()),
        vector_distance(pose.left_knee.to_xyz(), pose.right_knee.to_xyz()),

        vector_distance(pose.left_wrist.to_xyz(), pose.right_wrist.to_xyz()),
        vector_distance(pose.left_ankle.to_xyz(), pose.right_ankle.to_xyz()),

        # Body bent direction.
        vector_distance(
            lerp_vector_3d(pose.left_wrist.to_xyz(), pose.left_ankle.to_xyz(), 0.5),
            pose.left_hip.to_xyz()
        ),

        vector_distance(
            lerp_vector_3d(pose.right_wrist.to_xyz(), pose.right_ankle.to_xyz(), 0.5),
            pose.right_hip.to_xyz()
        )
    ])

    return embedding
