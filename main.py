import open3d as o3d
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np


def build_knn_graph(pos, k):
    # Use sklearn for KNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(pos.numpy())
    distances, indices = nbrs.kneighbors(pos.numpy())

    # Create edge_index
    edge_list = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:
                edge_list.append([i, neighbor])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def load_point_cloud_to_graph(path, k=16):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(path)

    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    pos = torch.tensor(np.asarray(pcd.points), dtype=torch.float)

    # Compute normals
    if len(pcd.normals) == 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        pcd.orient_normals_consistent_tangent_plane(k=k)

    normals = torch.tensor(np.asarray(pcd.normals), dtype=torch.float)

    # Create per point feature vector
    x = torch.cat([pos, normals], dim=1)

    edge_index = build_knn_graph(pos, k=k)

    # edge attributes
    row, col = edge_index
    edge_attr = pos[row] - pos[col]

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    return data


print(load_point_cloud_to_graph(o3d.data.PLYPointCloud().path))
