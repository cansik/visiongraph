import argparse
from argparse import _ArgumentGroup
from functools import partial
from typing import Union

from visiongraph.estimator.spatial.pose.AEPoseEstimator import AEPoseEstimator, AEPoseConfig
from visiongraph.estimator.spatial.pose.MediaPipePoseEstimator import MediaPipePoseEstimator, PoseModelComplexity
from visiongraph.estimator.spatial.pose.MoveNetPoseEstimator import MoveNetPoseEstimator, MoveNetConfig
from visiongraph.estimator.spatial.pose.OpenPoseEstimator import OpenPoseEstimator, OpenPoseConfig
from visiongraph.util.ArgUtils import add_step_choice_argument

PoseEstimators = {
    "mediapipe": partial(MediaPipePoseEstimator.create, PoseModelComplexity.Normal),
    "movenet": partial(MoveNetPoseEstimator.create, MoveNetConfig.MoveNet_MultiPose_256x320_FP32),
    "openpose": partial(OpenPoseEstimator.create, OpenPoseConfig.LightWeightOpenPose_FP32),
    "aepose": partial(AEPoseEstimator.create, AEPoseConfig.EfficientHRNet_288_FP32),
}


def add_pose_estimation_step_choices(parser: Union[argparse.ArgumentParser, _ArgumentGroup],
                                     default: int = 0, add_params: bool = False):
    add_step_choice_argument(parser, PoseEstimators, "--pose-estimator", help="Pose estimator",
                             default=default, add_params=add_params)
