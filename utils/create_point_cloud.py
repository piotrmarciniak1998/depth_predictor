import open3d as o3d
import numpy as np
import os
from pathlib import Path


DATASET = "1_60_circle"
NUMBER_OF_PREDICTIONS = None


def display_point_cloud():
    abs_path = os.path.abspath("../")
    predictions_path = f"{abs_path}/predictions/{DATASET}/output"
    input_names = [path.name for path in Path(predictions_path).iterdir()]
    if type(NUMBER_OF_PREDICTIONS) is int:
        input_names = input_names[:NUMBER_OF_PREDICTIONS]

    for input_name in input_names:
        print(input_name)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=320, height=240,
                                                             fx=400, fy=400,
                                                             cx=160, cy=120)
        depth = o3d.io.read_image(f"{predictions_path}/{input_name}")
        depth = np.asarray(depth)
        depth = (depth[:, :, 0] / 3 + depth[:, :, 1] / 3 + depth[:, :, 2] / 3)
        depth = depth.astype(np.uint16)
        depth = depth[:, :, np.newaxis]
        depth = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_intrinsic)
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)), center=(0, 0, 0))
        o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    display_point_cloud()
