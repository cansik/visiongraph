import argparse

import visiongraph as vg

if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisionGraph Example", description="Undistortion Pipeline")
    vg.VisionGraph.add_params(parser)

    parser.add_argument("--calibration", type=str, default="calibration.json",
                        help="Path where the calibration is stored.")

    args = parser.parse_args()

    intrinsics = vg.CameraIntrinsics.load(args.calibration)

    graph = (
        vg.create_graph(name="Undistortion Example", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .then(vg.ImagePreview("input"))
        .then(vg.UndistortionCalculator(intrinsics))
        .then(vg.ImagePreview("corrected"))
        .build()
    )
    graph.configure(args)
    graph.open()
