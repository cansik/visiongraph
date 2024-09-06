import argparse

from visiongraph import vg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisionGraph Example", description="LowLight Pipeline")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    graph = (
        vg.create_graph(name="LowLight Example", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .then(vg.ImagePreview("input"))
        .then(vg.MBLLENEstimator.create())
        .then(vg.ImagePreview("corrected"))
        .build()
    )
    graph.configure(args)
    graph.open()
