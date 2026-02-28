# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool

class SimpleGNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, n_layers=3, out_dim=1, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_dim = node_in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch, global_feats):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_all = torch.cat([x_mean, x_max, global_feats], dim=1)
        out = self.head(x_all)
        return out.squeeze(-1)
