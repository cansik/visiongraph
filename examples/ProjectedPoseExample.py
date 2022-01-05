import argparse
import logging
from argparse import ArgumentParser
from typing import Optional, Callable, List

import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import TriangleMesh
from open3d.cpu.pybind.visualization import rendering
from open3d.visualization import gui

from visiongraph.BaseGraph import BaseGraph
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.input import add_input_step_choices, RealSenseInput
from visiongraph.input.BaseInput import BaseInput
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util.LoggingUtils import add_logging_parameter

MIN_SCORE = 0.5


class ProjectedPoseExample(BaseGraph):

    def __init__(self, input: BaseInput, pose_network: PoseEstimator, multi_threaded: bool = True, deamon: bool = True):
        super().__init__(multi_threaded, deamon)
        self.input = input
        self.network = pose_network

        self.add_nodes(self.input, self.network)
        self.on_result_ready: Optional[Callable[[List[PoseLandmarkResult]], None]] = None

        self.use_projection = True

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.estimate(frame)
        rs: RealSenseInput = self.input

        for pose in results:
            for i, lm in enumerate(pose.landmarks):
                if lm.t < MIN_SCORE:
                    pose.landmarks.x[i] = 0
                    pose.landmarks.y[i] = 0
                    pose.landmarks.z[i] = 0
                    continue

                if self.use_projection:
                    p = rs.pixel_to_point(lm.x, lm.y)
                    pose.landmarks.x[i] = p.x
                    pose.landmarks.y[i] = p.y
                    pose.landmarks.z[i] = p.z
                else:
                    depth = rs.distance(lm.x, lm.y)
                    pose.landmarks.z[i] = depth

        for result in results:
            result.annotate(frame)

        if self.on_result_ready is not None:
            self.on_result_ready(results)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass


class MainWindow:
    def __init__(self, pipeline: ProjectedPoseExample):
        self.pipeline = pipeline

        self.vis = o3d.visualization.O3DVisualizer("Projected Point Example", 1024, 768)
        self.vis.show_skybox(False)
        self.vis.show_axes = True
        self.vis.show_ground = True
        self.vis.show_settings = True
        self.vis.point_size = 10

        self.vis.set_on_close(self._on_close)

        self.camera_geometry: TriangleMesh = o3d.geometry.TriangleMesh.create_box(0.4, 0.2, 0.2)
        self.camera_geometry.translate((-0.2, -0.1, -0.1))

        self.vis.add_geometry("realsense", self.camera_geometry)

        self.pose_cloud: Optional[o3d.geometry.PointCloud] = None
        self.lines: Optional[o3d.geometry.LineSet] = None

        gui.Application.instance.add_window(self.vis)

        # hook to events
        self.pipeline.on_result_ready = self.on_result_ready

        self._first_run = True

    def _on_close(self):
        gui.Application.instance.quit()

    def on_result_ready(self, results: List[PoseLandmarkResult]):
        pose_detected = False

        if len(results) > 0:
            # update points of pointcloud
            pose = results[0]

            if self.pose_cloud is None:
                self.pose_cloud = self._make_point_cloud(len(pose.landmarks), (0, 0, 0), 1)

            lm_positions = pose.landmarks.to_xyz()
            size = len(lm_positions)
            points = np.concatenate((lm_positions.x.reshape(size, 1) * -1,
                                     lm_positions.y.reshape(size, 1) * -1,
                                     lm_positions.z.reshape(size, 1)),
                                    axis=1)
            self.pose_cloud.points = o3d.utility.Vector3dVector(points)

            connections = [line for line in pose.connections
                           if pose.landmarks.t[line[0]] >= MIN_SCORE and pose.landmarks.t[line[1]] >= MIN_SCORE]

            # create connection lines
            self.lines = o3d.geometry.LineSet().create_from_point_cloud_correspondences(self.pose_cloud,
                                                                                        self.pose_cloud,
                                                                                        connections)

            pose_detected = True

        def update():
            if self._first_run and pose_detected:
                self.vis.add_geometry("pose", self.pose_cloud)
                self.vis.add_geometry("lines", self.lines)
                self.vis.reset_camera_to_default()
                self._first_run = False

            if pose_detected:
                update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                                rendering.Scene.UPDATE_COLORS_FLAG)
                # not working atm
                # self.vis.update_geometry("pose", self.pose_cloud, update_flags)
                self.vis.remove_geometry("pose")
                self.vis.add_geometry("pose", self.pose_cloud)

                self.vis.remove_geometry("lines")
                self.vis.add_geometry("lines", self.lines)

        gui.Application.instance.post_to_main_thread(self.vis, update)

    def _make_point_cloud(self, npts, center, radius):
        pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud


def main():
    pipeline = ProjectedPoseExample(args.input(), args.pose_estimator(), multi_threaded=True)
    pipeline.configure(args)
    pipeline.open()

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = MainWindow(pipeline)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Estimation Example", description="Example Pipeline")
    add_logging_parameter(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group, default=1)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group, default=3)

    args = parser.parse_args()

    if args.input is not RealSenseInput:
        logging.error("This example only runs with a RealSense Input")
        exit(1)

    args.depth = True
    main()
