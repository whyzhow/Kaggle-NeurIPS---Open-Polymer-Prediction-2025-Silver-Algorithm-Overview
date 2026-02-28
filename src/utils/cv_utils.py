# src/utils/cv_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Tuple, List
from sklearn.base import clone
from tqdm import tqdm

def get_group_kfold(df: pd.DataFrame, group_col: str, n_splits: int=5, random_state: int=42):
    """Return list of (train_idx, val_idx) pairs for GroupKFold based on group_col."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    X = np.zeros(len(df))
    y = df.index.values
    splits = []
    for tr_idx, val_idx in gkf.split(X, y, groups):
        splits.append((tr_idx, val_idx))
    return splits

def make_group_oof(estimator, X: pd.DataFrame, y: pd.Series, groups: np.ndarray, splits: List[Tuple], fit_params=None):
    """
    estimator: sklearn-like estimator with fit/predict
    splits: list of (train_idx, val_idx)
    """
    oof = np.zeros(len(y))
    val_idxs = []
    for fold, (tr_idx, val_idx) in enumerate(splits):
        est = clone(estimator)
        if fit_params:
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx], **fit_params)
        else:
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = est.predict(X.iloc[val_idx])
        oof[val_idx] = preds
        val_idxs.extend(val_idx.tolist())
    return oof
