import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk

from visiongraph.VisionGraphBuilder import create_graph, custom
from visiongraph.input.OakDInput import OakDInput


def main():
    rr.init("rerun_example_app")

    rr.disable_timeline("0")
    rr.spawn()

    oak_d = OakDInput()

    def on_frame(frame: np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("image", rr.Image(frame))

        rr.log("ir", rr.Image(oak_d.ir_frame))
        rr.log("depth", rr.Image(oak_d.depth_map))

    create_graph(name="VisionGraph", input_node=oak_d, handle_signals=True) \
        .then(custom(on_frame)) \
        .open()


if __name__ == "__main__":
    main()
