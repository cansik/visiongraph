import argparse
import logging
from argparse import _ArgumentGroup
from functools import partial
from typing import Union

from visiongraph.util.ArgUtils import add_step_choice_argument

PoseEstimators = {}

# setup optional pose estimators
try:
    from visiongraph.estimator.spatial.pose.MediaPipePoseEstimator import MediaPipePoseEstimator, PoseModelComplexity

    PoseEstimators["mediapipe"] = partial(MediaPipePoseEstimator.create, PoseModelComplexity.Normal)
    PoseEstimators["mediapipe-light"] = partial(MediaPipePoseEstimator.create, PoseModelComplexity.Light)
    PoseEstimators["mediapipe-heavy"] = partial(MediaPipePoseEstimator.create, PoseModelComplexity.Heavy)
except ImportError as ex:
    logging.info(f"MediaPipe not installed: {ex}")

try:
    from visiongraph.estimator.spatial.pose.MoveNetPoseEstimator import MoveNetPoseEstimator, MoveNetConfig

    PoseEstimators["movenet"] = partial(MoveNetPoseEstimator.create, MoveNetConfig.MoveNet_MultiPose_256x320_FP32)
    PoseEstimators["movenet-192"] = partial(MoveNetPoseEstimator.create, MoveNetConfig.MoveNet_MultiPose_192x256_FP32)
    PoseEstimators["movenet-single"] = partial(MoveNetPoseEstimator.create, MoveNetConfig.MoveNet_Single_Thunder_FP32)
except ImportError as ex:
    logging.info(f"MoveNet not installed: {ex}")

try:
    from visiongraph.estimator.spatial.pose.LiteHRNetEstimator import LiteHRNetPoseEstimator, LiteHRNetConfig

    prefix = "lite-hrnet"
    PoseEstimators[f"{prefix}-fp16"] = partial(LiteHRNetPoseEstimator.create,
                                               LiteHRNetConfig.LiteHRNet_30_COCO_384x288_FP16)
    PoseEstimators[f"{prefix}"] = partial(LiteHRNetPoseEstimator.create, LiteHRNetConfig.LiteHRNet_30_COCO_384x288_FP16)
    PoseEstimators[f"{prefix}-fast"] = partial(LiteHRNetPoseEstimator.create,
                                               LiteHRNetConfig.LiteHRNet_18_COCO_256x192_FP32)
except ImportError as ex:
    logging.info(f"MoveNet not installed: {ex}")

try:
    from visiongraph.estimator.spatial.pose.AEPoseEstimator import AEPoseEstimator, AEPoseConfig
    from visiongraph.estimator.spatial.pose.OpenPoseEstimator import OpenPoseEstimator, OpenPoseConfig
    from visiongraph.estimator.spatial.pose.MobileNetV2PoseEstimator import MobileNetV2PoseEstimator, \
        MobileNetV2PoseEstimatorConfig
    from visiongraph.estimator.spatial.pose.EfficientPoseEstimator import EfficientPoseEstimator, \
        EfficientPoseEstimatorConfig
    from visiongraph.estimator.spatial.pose.LitePoseEstimator import LitePoseEstimator, LitePoseEstimatorConfig

    PoseEstimators["openpose"] = partial(OpenPoseEstimator.create, OpenPoseConfig.LightWeightOpenPose_FP32)
    PoseEstimators["openpose-int8"] = partial(OpenPoseEstimator.create, OpenPoseConfig.LightWeightOpenPose_INT8)
    PoseEstimators["openpose-fp16"] = partial(OpenPoseEstimator.create, OpenPoseConfig.LightWeightOpenPose_FP16)

    PoseEstimators["aepose"] = partial(AEPoseEstimator.create, AEPoseConfig.EfficientHRNet_288_FP32)
    PoseEstimators["aepose-288-fp16"] = partial(AEPoseEstimator.create, AEPoseConfig.EfficientHRNet_288_FP16)
    PoseEstimators["aepose-448-fp32"] = partial(AEPoseEstimator.create, AEPoseConfig.EfficientHRNet_448_FP32)

    PoseEstimators["mobilenet"] = partial(MobileNetV2PoseEstimator.create,
                                          MobileNetV2PoseEstimatorConfig.MNV2PE_1_4_224_FP32)

    PoseEstimators["efficient-pose"] = partial(EfficientPoseEstimator.create,
                                               EfficientPoseEstimatorConfig.EFFICIENT_POSE_I_FP32)
    PoseEstimators["efficient-pose-lite"] = partial(EfficientPoseEstimator.create,
                                                    EfficientPoseEstimatorConfig.EFFICIENT_POSE_I_LITE_FP32)
    PoseEstimators["efficient-pose-rt"] = partial(EfficientPoseEstimator.create,
                                                  EfficientPoseEstimatorConfig.EFFICIENT_POSE_RT_FP32)

    PoseEstimators["litepose"] = partial(LitePoseEstimator.create,
                                                  LitePoseEstimatorConfig.LitePose_S_COCO_FP32)
except ImportError as ex:
    logging.info(f"OpenVino not installed: {ex}")

try:
    from visiongraph.estimator.spatial.pose.MobileHumanPoseEstimator import MobileHumanPoseEstimator

    PoseEstimators["mobile-human-pose"] = MobileHumanPoseEstimator
except ImportError as ex:
    logging.info(f"OpenVino or ONNX not installed: {ex}")


def add_pose_estimation_step_choices(parser: Union[argparse.ArgumentParser, _ArgumentGroup],
                                     default: Union[int, str] = 0, add_params: bool = False):
    add_step_choice_argument(parser, PoseEstimators, "--pose-estimator", help="Pose estimator",
                             default=default, add_params=add_params)
