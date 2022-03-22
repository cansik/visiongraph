import argparse
from argparse import ArgumentParser
from typing import Optional

import cv2
import numpy as np

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.inpaint.GMCNNInpainter import GMCNNInpainter
from visiongraph.input import add_input_step_choices
from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.LoggingUtils import add_logging_parameter


class InpaintExample(BaseGraph):

    def __init__(self, input: BaseInput):
        super().__init__()
        self.input = input
        self.network = GMCNNInpainter.create()

        self.mask: Optional[np.ndarray] = None

        self.add_nodes(self.input, self.network)

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        result = self.network.inpaint(frame, self.mask)
        preview = np.hstack((frame, result.output))

        cv2.imshow("Inpaint Example", preview)
        if cv2.waitKey(0) & 0xFF == 27:
            self.close()

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass

    def configure(self, args: argparse.Namespace):
        super().configure(args)
        self.mask = cv2.imread(args.mask)


def main():
    pipeline = InpaintExample(args.input())
    pipeline.configure(args)
    pipeline.open()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inpaint Example", description="Example Pipeline")
    add_logging_parameter(parser)
    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group)

    parser.add_argument("--mask", required=True, type=str, help="Inpainting mask to use.")

    args = parser.parse_args()

    main()
