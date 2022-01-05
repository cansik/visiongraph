# simplest vision graph example
import visiongraph as vg

graph = vg.VisionGraph(name="Object Detection", input=vg.VideoCaptureInput(1), multi_threaded=False,
                       ssd=vg.ChainEstimator(vg.SSDDetector.create(), vg.ObjectDetectionTracker()))
graph.open()
