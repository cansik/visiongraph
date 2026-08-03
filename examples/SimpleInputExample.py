from visiongraph import vg


def main():
    (
        vg.create_graph(name="VisionGraph", input_node=vg.VideoCaptureInput(), handle_signals=True)
        .then(vg.ImagePreview())
        .open()
    )


if __name__ == "__main__":
    main()
