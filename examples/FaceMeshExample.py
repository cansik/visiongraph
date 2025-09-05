import argparse

from visiongraph import vg


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser("FaceMesh Example", description="Detect face meshes on images.")
    vg.VisionGraph.add_params(parser)
    args = parser.parse_args()

    # define graph
    graph = (
        vg.create_graph(name="FaceMesh Example", input_node=args.input(), handle_signals=True)
        # run detection and pass image through for annotation
        .apply(
            image=vg.passthrough(),
            face_mesh=vg.sequence(
                vg.MediaPipeFaceMeshEstimator(
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
            ),
        )
        # annotate result
        .then(vg.ResultAnnotator(), vg.ImagePreview("Preview"))
        .build()
    )
    graph.configure(args)

    # start graph
    graph.open()


if __name__ == "__main__":
    main()
