import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
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


class GNNLayer(MessagePassing):
    """Graph Neural Network layer using message passing.

    This layer aggregates information from neighboring nodes to update node features.
    """

    def __init__(self, in_channels, out_channels):
        # Use mean aggregation to combine messages from neighbors
        super().__init__(aggr="mean")
        # MLP processes concatenated node and edge features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        # Propagate messages along edges to update node features
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        # Concatenate neighbor features with edge features
        msg = torch.cat([x_j, edge_attr], dim=-1)
        # Transform through MLP
        return self.mlp(msg)


class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder for point cloud feature extraction.

    Stacks multiple GNN layers
    """

    def __init__(self, input_dim=6, hidden_dim=64, out_dim=32):
        super().__init__()
        self.layer1 = GNNLayer(input_dim + 3, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim + 3, hidden_dim)
        self.layer3 = GNNLayer(hidden_dim + 3, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First GNN layer + activation
        x = self.layer1(x, edge_index, edge_attr)
        x = F.relu(x)

        # Second GNN layer + activation
        x = self.layer2(x, edge_index, edge_attr)
        x = F.relu(x)

        # Third GNN layer (no activation - final embedding)
        x = self.layer3(x, edge_index, edge_attr)

        return x
