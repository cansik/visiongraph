# simplest vision graph example
import visiongraph as vg

if __name__ == "__main__":
    graph = vg.VisionGraph(name="Object Detection", input=vg.VideoCaptureInput(1),
                           ssd=vg.ChainEstimator(vg.SSDDetector.create(), vg.ObjectDetectionTracker()))
    graph.open()
