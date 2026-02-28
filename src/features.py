# src/features.py
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load
from tqdm import tqdm
from typing import List

from config import FEATS_DIR
os.makedirs(FEATS_DIR, exist_ok=True)

def calc_morgan_fp(smiles: List[str], n_bits=2048, radius=2, cache_path=None):
    """Compute Morgan fingerprints (bit vectors) and optionally cache to disk."""
    fps = []
    for s in tqdm(smiles, desc="morgan"):
        try:
            mol = Chem.MolFromSmiles(s)
            v = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((1,), dtype=np.uint8)
            # convert to numpy array
            a = np.zeros((n_bits,), dtype=np.int8)
            for i in range(n_bits):
                a[i] = int(v.GetBit(i))
            fps.append(a)
        except Exception:
            fps.append(np.zeros((n_bits,), dtype=np.int8))
    fps = np.vstack(fps)
    if cache_path:
        dump(fps, cache_path)
    return fps

def reduce_fp(fp_array, n_components=200, cache_path=None):
    """TruncatedSVD on fingerprint matrix (sparse -> dense reduction)."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_red = svd.fit_transform(fp_array)
    if cache_path:
        dump(svd, cache_path + ".svd")
        pd.DataFrame(X_red).to_parquet(cache_path + ".parquet", index=False)
    return X_red

def compute_basic_descriptors(smiles: List[str]):
    """Compute a small set of RDKit descriptors (mass, TPSA, logP)"""
    from rdkit.Chem import Descriptors
    feats = []
    for s in tqdm(smiles, desc="descriptors"):
        try:
            mol = Chem.MolFromSmiles(s)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            logp = Descriptors.MolLogP(mol)
            feats.append([mw, tpsa, logp])
        except Exception:
            feats.append([0.0, 0.0, 0.0])
    return np.array(feats)

def build_feature_table(df, smiles_col="c_smiles", n_bits=2048, n_svd=200, prefix="train"):
    """Compute and cache fingerprints + descriptors; return DataFrame of features."""
    cache_fp = os.path.join(FEATS_DIR, f"{prefix}_fp.joblib")
    cache_svd = os.path.join(FEATS_DIR, f"{prefix}_fp_svd")
    cache_desc = os.path.join(FEATS_DIR, f"{prefix}_desc.parquet")
    if os.path.exists(cache_svd + ".parquet"):
        fp_red = pd.read_parquet(cache_svd + ".parquet").values
    else:
        fps = calc_morgan_fp(df[smiles_col].fillna("").tolist(), n_bits=n_bits, cache_path=cache_fp)
        fp_red = reduce_fp(fps, n_components=min(n_svd, fps.shape[1] - 1), cache_path=cache_svd)
    if os.path.exists(cache_desc):
        desc = pd.read_parquet(cache_desc).values
    else:
        desc = compute_basic_descriptors(df[smiles_col].fillna("").tolist())
        pd.DataFrame(desc, columns=["MolWt", "TPSA", "LogP"]).to_parquet(cache_desc, index=False)
    feat = np.hstack([fp_red, desc])
    feat_df = pd.DataFrame(feat, index=df.index)
    return feat_df
