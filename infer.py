import torch
import pandas as pd
from models import PolymerGNN
from dataset import smiles_to_graph
from config import CFG
from torch_geometric.loader import DataLoader
import numpy as np


def inference():
    df = pd.read_csv("test.csv")
    graphs = []

    for i in range(len(df)):
        g = smiles_to_graph(df.iloc[i]["SMILES"])
        graphs.append(g)

    loader = DataLoader(graphs, batch_size=CFG.batch_size)

    preds = np.zeros((len(df), CFG.num_targets))

    for fold in range(CFG.n_folds):
        model = PolymerGNN(num_targets=CFG.num_targets).to(CFG.device)
        model.load_state_dict(torch.load(f"checkpoints/fold_{fold}.pth"))
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(CFG.device)
                p = model(batch).cpu().numpy()
                fold_preds.append(p)

        fold_preds = np.vstack(fold_preds)
        preds += fold_preds

    preds /= CFG.n_folds

    submission = pd.DataFrame(preds, columns=[f"target_{i}" for i in range(CFG.num_targets)])
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    inference()