import argparse
from argparse import ArgumentParser

import cv2

from visiongraph.BaseGraph import BaseGraph
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter
from visiongraph.util.TimeUtils import FPSTracer


class InputExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.add_nodes(self.input)

        self.fps_tracer = FPSTracer()

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        if isinstance(self.input, BaseDepthInput):
            depth = self.input.distance(0.5, 0.5)
            print(f"{depth:.2f}m")

        self.fps_tracer.update()
        cv2.putText(frame, "FPS: %.0f" % self.fps_tracer.smooth_fps,
                    (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Input Example", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


def main():
    pipeline = InputExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Input Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    args = parser.parse_args()

    main()
