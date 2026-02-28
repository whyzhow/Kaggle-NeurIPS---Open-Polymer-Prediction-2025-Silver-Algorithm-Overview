# src/train.py
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

from config import SEED, N_SPLITS, MODELS_DIR, FEATS_DIR, OOF_DIR, DEVICE
from src.utils.smiles_utils import batch_canonicalize
from src.utils.cv_utils import get_group_kfold, make_group_oof
from src.features import build_feature_table
from src.stacking.optuna_stack import optimize_weights

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def create_demo_data(path="dataset.csv", n=200):
    data = {
        "id": list(range(n)),
        "smiles": ["CCO", "CC", "CCC", "c1ccccc1", "C1CCCCC1"] * (n//5),
        "target": list(np.random.RandomState(SEED).uniform(0,1,size=n))
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path

def main(args):
    seed_everything(args.seed)
    # Load or create data
    if args.demo:
        data_path = create_demo_data("dataset_demo.csv", n=200)
    else:
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found. Use --demo to create sample data.")
    df = pd.read_csv(data_path)
    # canonicalize SMILES
    c_smiles, failed = batch_canonicalize(df['smiles'].fillna("").tolist(), save_failed=os.path.join(OOF_DIR,"failed_smiles.csv"))
    df['c_smiles'] = c_smiles
    # For groups: use canonical smiles as group key
    df['group'] = df['c_smiles'].fillna("NA")
    # compute features (cached)
    feat_df = build_feature_table(df, smiles_col="c_smiles", prefix="demo" if args.demo else "train")
    # align
    X = feat_df
    y = df['target']
    # CV splits
    splits = get_group_kfold(df, group_col='group', n_splits=args.n_splits)
    # Train CatBoost (as example)
    cb = CatBoostRegressor(loss_function="MAE", iterations=2000, learning_rate=0.03, depth=6, random_seed=args.seed, verbose=100)
    # OOF for CatBoost
    oof_cb = np.zeros(len(df))
    for fold, (tr_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        pool_tr = Pool(X_tr, y_tr)
        pool_val = Pool(X_val, y_val)
        cb.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=50, use_best_model=True)
        oof_cb[val_idx] = cb.predict(X_val)
    dump(cb, os.path.join(MODELS_DIR, "catboost_demo.cbm"))
    pd.DataFrame({"oof_cb": oof_cb}).to_csv(os.path.join(OOF_DIR, "oof_catboost.csv"), index=False)
    # XGBoost model
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=args.seed)
    oof_xgb = np.zeros(len(df))
    for fold, (tr_idx, val_idx) in enumerate(splits):
        xgb.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], early_stopping_rounds=50, verbose=False)
        oof_xgb[val_idx] = xgb.predict(X.iloc[val_idx])
    dump(xgb, os.path.join(MODELS_DIR, "xgb_demo.joblib"))
    pd.DataFrame({"oof_xgb": oof_xgb}).to_csv(os.path.join(OOF_DIR, "oof_xgb.csv"), index=False)
    # Simple meta: train Ridge on OOFs
    meta_X = np.vstack([oof_cb, oof_xgb]).T
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, y)
    dump(meta, os.path.join(MODELS_DIR, "meta_ridge.joblib"))
    # Evaluate
    pred_meta = meta.predict(meta_X)
    print("MAE catboost:", mean_absolute_error(y, oof_cb))
    print("MAE xgb:", mean_absolute_error(y, oof_xgb))
    print("MAE meta:", mean_absolute_error(y, pred_meta))
    # Optional: use optuna to find best weights between models
    oof_preds = np.vstack([oof_cb, oof_xgb])
    best_w = optimize_weights(oof_preds, y.values, n_trials=50, seed=args.seed)
    print("best weights:", best_w)
    ensemble_pred = (best_w.reshape(-1,1) * oof_preds).sum(axis=0)
    print("MAE weighted ensemble:", mean_absolute_error(y, ensemble_pred))
    # Save OOF and params
    pd.DataFrame({"ensemble_pred": ensemble_pred}).to_csv(os.path.join(OOF_DIR, "oof_ensemble.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "params.json"), "w") as f:
        json.dump({"seed": args.seed, "n_splits": args.n_splits, "best_weights": best_w.tolist()}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Run demo on synthetic small dataset")
    args = parser.parse_args()
    main(args)
