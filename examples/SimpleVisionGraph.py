# simplest vision graph example
import visiongraph as vg

graph = vg.VisionGraph({"ssd": vg.ChainEstimator(vg.SSDDetector.create(), vg.ObjectDetectionTracker())}, name="Object Detection", input=vg.VideoCaptureInput(), multi_threaded=False)
graph.open()
