import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class PolymerGNN(nn.Module):
    def __init__(self, atom_dim=8, hidden_dim=128, num_layers=3, num_targets=5, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(atom_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.attn = GATConv(hidden_dim, hidden_dim // 4, heads=4)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_targets)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = torch.relu(bn(conv(x, edge_index)))

        x = self.attn(x, edge_index)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        h = torch.cat([mean_pool, max_pool], dim=1)

        outputs = [head(h) for head in self.heads]
        return torch.cat(outputs, dim=1)
    
class FingerprintMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_targets=5, dropout=0.3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets)
        )

    def forward(self, x):
        return self.model(x)