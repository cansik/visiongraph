import argparse
from argparse import ArgumentParser
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from visiongraph.Pipeline import Pipeline
from visiongraph.input import add_input_step_choices, VideoCaptureInput
from visiongraph.input.BaseInput import BaseInput
from visiongraph.recorder.CV2VideoRecorder import CV2VideoRecorder
from visiongraph.signal.OneEuroFilter import OneEuroFilter
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class RGBDSmoother(Pipeline):

    def __init__(self, input: BaseInput, multi_threaded: bool = True, deamon: bool = True):
        super().__init__(multi_threaded, deamon)
        self.input: VideoCaptureInput = input

        self.input.loop = False
        self.input.fps_lock = False

        self.filter: Optional[OneEuroFilter] = None
        self.output_path = None

        self.recorder: Optional[CV2VideoRecorder] = None
        self.fps_tracer = FPSTracer()

        self.add_nodes(self.input)

        self.progress_bar: Optional[tqdm] = None

    def _init(self):
        super()._init()

        self.progress_bar = tqdm(desc="smoothing", total=self.input.frame_count)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            self.close()
            return

        # extract depth only
        w, h = int(self.input.width // 2), int(self.input.height)
        depth_frame = np.copy(frame[0:h, 0:w])
        rgb_frame = frame[0:h, w:w + w]

        # extract hue
        bgr_frame = cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR)
        hue_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_RGB2HSV_FULL)
        depth = hue_frame[:, :, 0].astype("float32") / 255.0

        # late init
        if self.filter is None:
            self.filter = OneEuroFilter(depth, min_cutoff=0.5, beta=0.001)
            smooth_depth = depth

            if self.output_path is not None:
                self.recorder = CV2VideoRecorder(self.input.width, self.input.height, self.output_path, self.input.fps)
                self.recorder.open()
        else:
            smooth_depth = self.filter(depth)

        # recreate hue
        hue_values = np.round(smooth_depth * 255).astype("uint8")
        hue_values = hue_values.reshape((*hue_values.shape, 1))
        smooth_hue = np.concatenate((hue_values, hue_frame[:, :, 1:]), axis=2)

        smooth_bgr = cv2.cvtColor(smooth_hue, cv2.COLOR_HSV2RGB_FULL)
        smooth_rgb = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGRA2RGB)

        smooth_rgbd = np.hstack((smooth_rgb, rgb_frame))

        if self.recorder is not None:
            self.recorder.add_image(smooth_rgbd)

        stacked = np.hstack((smooth_rgb, depth_frame, rgb_frame))

        self.fps_tracer.update()
        cv2.putText(stacked, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                    (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("RGB-D Smoother (Smooth, Original, RGB)", stacked)
        if cv2.waitKey(15) & 0xFF == 27:
            self.close()

        self.progress_bar.update()

    def _release(self):
        super()._release()

        if self.recorder is not None:
            self.recorder.close()

    def configure(self, args: argparse.Namespace):
        super().configure(args)
        self.output_path = args.output

    @staticmethod
    def get_gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = RGBDSmoother(args.input(), multi_threaded=False)
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RGB-D Smoother", description="Smoothes RGB-D videos")
    add_logging_parameter(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    parser.add_argument("--output", type=str, default=None, help="Output path if conversion should be stored.")

    args = parser.parse_args()

    main()
