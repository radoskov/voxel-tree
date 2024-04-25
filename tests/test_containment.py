from random import random

import numpy as np
import open3d as o3d
from sampled_boxes import SampledAABB, SampledBBox
from scipy.spatial.transform import Rotation


def test_normals():
    box = SampledAABB(np.array([0, 0, 0]), np.array([1, 1, 1]))
    face_a = np.cross(box.corners[2] - box.corners[0], box.corners[1] - box.corners[0])
    face_b = np.cross(box.corners[1] - box.corners[0], box.corners[4] - box.corners[0])
    face_c = np.cross(box.corners[4] - box.corners[0], box.corners[2] - box.corners[0])




def test_containment():
    # box = SampledAABB(np.array([0, 0, 0]), np.array([1, 1, 1]))
    # box_points = o3d.utility.Vector3dVector(box.corners)
    box_points = o3d.utility.Vector3dVector(np.random.randn(8, 3))

    obox = o3d.geometry.OrientedBoundingBox.create_from_points(box_points)
    obox.color = np.array([1, 0, 1])

    r = Rotation.from_euler('xyz', [random() * 360, random() * 360, random() * 360], degrees=True)
    obox = obox.rotate(r.as_matrix())
    obox = obox.translate(np.random.randn(3))

    box = SampledBBox.from_o3d_obb(obox)

    random_points = np.random.randn(100000, 3)
    random_pcd = o3d.geometry.PointCloud()
    random_pcd.points = o3d.utility.Vector3dVector(random_points)

    idx_inside = obox.get_point_indices_within_bounding_box(random_pcd.points)
    # idx_inside = np.where([box.contains_point(p) for p in random_points])[0]
    # idx_inside = np.nonzero(box.contains_point(random_points))[0]

    idx_outside = np.setdiff1d(np.arange(len(random_points)), idx_inside)

    pcd_inside = random_pcd.select_by_index(idx_inside)
    pcd_outside = random_pcd.select_by_index(idx_inside, invert=True)

    pcd_inside.paint_uniform_color([1, 0, 0])
    pcd_outside.paint_uniform_color([0.1, 0.1, .1])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([axis, pcd_inside, pcd_outside, obox], window_name="test_containment")
    # o3d.visualization.draw_geometries([axis, pcd_inside, obox], window_name="test_containment")
    # o3d.visualization.draw_geometries([axis, pcd_outside, obox], window_name="test_containment")

    assert np.all([box.contains_point(p) for p in random_points[idx_inside]])
    assert np.all([not box.contains_point(p) for p in random_points[idx_outside]])


if __name__ == "__main__":
    test_containment()
