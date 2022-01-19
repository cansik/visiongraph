import argparse
import logging
from argparse import ArgumentParser
from typing import Optional, Callable, List

import numpy as np
import open3d as o3d
import vector
from open3d.cpu.pybind.geometry import TriangleMesh
from open3d.cpu.pybind.visualization import rendering
from open3d.visualization import gui

import visiongraph as vg
from visiongraph.estimator.spatial.pose import add_pose_estimation_step_choices
from visiongraph.input import add_input_step_choices

MIN_SCORE = 0.3


class ProjectedPoseExample(vg.BaseGraph):

    def __init__(self, input: vg.RealSenseInput, pose_network: vg.PoseEstimator):
        super().__init__(multi_threaded=True, deamon=True)
        self.input = input
        self.network = pose_network

        self.add_nodes(self.input, self.network)
        self.on_result_ready: Optional[Callable[[List[vg.PoseLandmarkResult]], None]] = None

        self.use_projection = True

    def _process(self):
        ts, frame = self.input.read()

        if frame is None:
            return

        results = self.network.process(frame)
        rs: vg.RealSenseInput = self.input

        translation_vector = vector.obj(x=0, y=-args.translation_y, z=0)

        for pose in results:
            for i, lm in enumerate(pose.landmarks):
                if lm.t < MIN_SCORE:
                    pose.landmarks.x[i] = 0
                    pose.landmarks.y[i] = 0
                    pose.landmarks.z[i] = 0
                    continue

                if self.use_projection:
                    p = rs.pixel_to_point(lm.x, lm.y)

                    # translate & rotate points
                    p = p.add(translation_vector)
                    p = p.rotateX(-np.radians(args.angle))

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
        self.camera_geometry.paint_uniform_color([0.1, 0.1, 0.7])
        self.camera_geometry.translate((-0.2, -0.1, -0.1))

        R = self.camera_geometry.get_rotation_matrix_from_xyz((np.radians(-args.angle), 0, 0))
        self.camera_geometry.rotate(R, center=(0, 0, 0))

        self.camera_geometry.translate((0, -args.translation_y, 0))

        self.vis.add_geometry("realsense", self.camera_geometry)

        self.pose_cloud: Optional[o3d.geometry.PointCloud] = None
        self.lines: Optional[o3d.geometry.LineSet] = None

        gui.Application.instance.add_window(self.vis)

        # hook to events
        self.pipeline.on_result_ready = self.on_result_ready

        self._first_run = True

    def _on_close(self):
        gui.Application.instance.quit()

    def on_result_ready(self, results: List[vg.PoseLandmarkResult]):
        pose_detected = False

        if len(results) > 0:
            # update points of pointcloud
            pose = results[0]

            if self.pose_cloud is None:
                self.pose_cloud = self._make_point_cloud(len(pose.landmarks), (0, 0, 0), 1)

            lm_positions = pose.landmarks.to_xyz()
            size = len(lm_positions)
            points = np.concatenate((lm_positions.x.reshape(size, 1), #* -1, # this is used if camera is not upside down
                                     lm_positions.y.reshape(size, 1), #* -1,
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

                if self.lines.has_lines():
                    self.vis.add_geometry("lines", self.lines)

                self.vis.reset_camera_to_default()
                self._first_run = False
            elif not self._first_run and pose_detected:
                update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                                rendering.Scene.UPDATE_COLORS_FLAG)
                # not working atm
                # self.vis.update_geometry("pose", self.pose_cloud, update_flags)
                self.vis.remove_geometry("pose")
                self.vis.add_geometry("pose", self.pose_cloud)

                if self.lines.has_lines():
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
    pipeline = ProjectedPoseExample(args.input(), args.pose_estimator())
    pipeline.configure(args)
    pipeline.open()

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = MainWindow(pipeline)
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pose Estimation Example", description="Example Pipeline")
    vg.add_logging_parameter(parser)

    input_group = parser.add_argument_group("input provider")
    add_input_step_choices(input_group, default=1)

    pose_group = parser.add_argument_group("pose estimator")
    add_pose_estimation_step_choices(pose_group, default=3)

    transform_group = parser.add_argument_group("camera transform")
    transform_group.add_argument("--angle", default=-30, type=float,
                                 help="Angle (degree) how much the camera is tilted.")
    transform_group.add_argument("--translation-y", default=-1.00, type=float,
                                 help="Distance (m) to translate the camera.")

    args = parser.parse_args()

    if args.input is not vg.RealSenseInput:
        logging.error("This example only runs with a RealSense Input")
        exit(1)

    args.depth = True
    main()
