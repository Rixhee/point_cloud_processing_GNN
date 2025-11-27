import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import networkx as nx

def load_pcd_as_o3d(path):
    """
    Load a nuScenes .pcd.bin LiDAR file as an Open3D PointCloud
    """
    # Load raw binary file
    scan = np.fromfile(path, dtype=np.float32)
    points = scan.reshape(-1, 5)[:, :3] # this just gives us the x, y, z terms, we don't really care about the other attributes rn for testing.

    # actual conversion to point cloud from our .bin data -> we want a pcd
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

    return point_cloud_o3d

def main():

    filename = "tested_pointclouds/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin"
    point_cloud_final = load_pcd_as_o3d(filename) # change the format.
    o3d.visualization.draw_geometries([point_cloud_final]) # actual visualization.

if __name__ == "__main__":
    main()
