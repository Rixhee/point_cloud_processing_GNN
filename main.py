import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
import numpy as np

##################################
# CREATING THE GRAPH.
##################################

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

    # Compute normals -- vectors for directions of surfaces.
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

##################################
# CREATING THE GNN.
##################################

class GNNLayer(MessagePassing):
    """Graph Neural Network layer using message passing.

    This layer aggregates information from neighboring nodes to update node features.
    """

    def __init__(self, in_channels, out_channels):
        # Use mean aggregation to combine messages from neighbors
        super().__init__(aggr="mean")
        # MLP processes concatenated node and edge features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
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
        self.layer1 = GNNLayer(input_dim, hidden_dim)
        self.layer2 = GNNLayer(hidden_dim, hidden_dim)
        self.layer3 = GNNLayer(hidden_dim, out_dim)


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
    
    
##################################
# CONTRASTIVE LEARNING 
##################################

def contrastive_cosine_loss(z1, z2, temperature=0.5):
    """ 
    Computes the contrastive loss between feature vectors.
    z1, z2 -> our given feature vectors
    temperature -> some arbitrary sensitivity that we have to difference (a.k.a tau).
    """

    # NORMALIZE THE FEATURE VECTORS.

    z1_norm = F.normalize(z1, dim=-1)  # FEATURE VECTOR 1
    z2_norm = F.normalize(z2, dim=-1)  # FEATURE VECTOR 2 

    # COSINE SIMILARITY MATRIX.

    # basically we just compute the similarity through the angle between them,
    # representing what vectors are similar and which are different.
    # S_{ij} = (z1_hat_i . z2_hat_j) / tau
    sim_matrix = torch.matmul(z1_norm, z2_norm.T) / temperature  # [N, N]
    positives = torch.diag(sim_matrix)    # Positive pairs are on the diagonal: S_{ii}

    # COMPUTING THE DENOMINATOR

    # we just look at the connection between the similarity between vector 1 and 2.
    # sums over all columns of the similarity matrix for every row.
    exp_sim = torch.exp(sim_matrix)  # e^{S_{ij}}
    denominator = exp_sim.sum(dim=1)  # sum_j e^{S_{ij}}

    # COMPUTE INFONCE LOSS -- actively pushing pairs closer and further away from eachother.
    # l_i = -log( exp(S_{ii}) / sum_j exp(S_{ij}) ) # mathematical notation.
    # every similar point should become 'closer' -- aka pointing more in the direction of the respective vector
    loss_per_point = -torch.log(torch.exp(positives) / denominator)

    # COMPUTE THE MEAN OVER EVERY POINT.
    loss = loss_per_point.mean()
    return loss

# Small augmentation for contrastive learning
def augment_point_cloud(data, jitter=0.01):
    pos = data.pos.clone()
    pos += torch.randn_like(pos) * jitter  # small random noise
    x = torch.cat([pos, data.x[:, 3:]], dim=1)  # keep normals
    row, col = data.edge_index
    edge_attr = pos[row] - pos[col]
    return Data(x=x, pos=pos, edge_index=data.edge_index, edge_attr=edge_attr)

# Instantiate encoder and optimizer
encoder = GNNEncoder(input_dim=6, hidden_dim=64, out_dim=32)
encoder.train()
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
temperature = 0.5
num_epochs = 100

# Load point cloud
data = load_point_cloud_to_graph(o3d.data.PLYPointCloud().path, k=16)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Two augmented views
    data1 = augment_point_cloud(data)
    data2 = augment_point_cloud(data)
    
    # Forward pass
    z1 = encoder(data1)
    z2 = encoder(data2)
    
    # Compute contrastive loss
    loss = contrastive_cosine_loss(z1, z2, temperature)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Contrastive Loss: {loss.item():.4f}")


    

