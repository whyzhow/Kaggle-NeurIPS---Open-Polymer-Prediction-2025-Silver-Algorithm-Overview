import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from models import PolymerGNN, FingerprintMLP
from dataset import smiles_to_graph, build_features
from config import CFG
import os


def train():

    df = pd.read_csv("train.csv")

    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(df)):

        print("Fold", fold)

        train_graphs = []
        val_graphs = []

        train_fp = []
        val_fp = []

        for i in trn_idx:
            g = smiles_to_graph(df.iloc[i]["SMILES"])
            g.y = torch.tensor(df.iloc[i][1:].values, dtype=torch.float)
            train_graphs.append(g)

            train_fp.append(build_features(df.iloc[i]["SMILES"]))

        for i in val_idx:
            g = smiles_to_graph(df.iloc[i]["SMILES"])
            g.y = torch.tensor(df.iloc[i][1:].values, dtype=torch.float)
            val_graphs.append(g)

            val_fp.append(build_features(df.iloc[i]["SMILES"]))

        train_loader = DataLoader(train_graphs, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=CFG.batch_size)

        fp_dim = len(train_fp[0])
        fp_model = FingerprintMLP(fp_dim, num_targets=CFG.num_targets).to(CFG.device)
        gnn_model = PolymerGNN(num_targets=CFG.num_targets).to(CFG.device)

        opt_fp = torch.optim.Adam(fp_model.parameters(), lr=CFG.lr)
        opt_gnn = torch.optim.Adam(gnn_model.parameters(), lr=CFG.lr)

        for epoch in range(CFG.epochs):

            gnn_model.train()
            fp_model.train()

            for batch in train_loader:
                batch = batch.to(CFG.device)

                gnn_pred = gnn_model(batch)
                loss_gnn = torch.nn.functional.mse_loss(gnn_pred, batch.y)

                opt_gnn.zero_grad()
                loss_gnn.backward()
                opt_gnn.step()

        torch.save(gnn_model.state_dict(), f"checkpoints/gnn_fold{fold}.pth")
        torch.save(fp_model.state_dict(), f"checkpoints/fp_fold{fold}.pth")