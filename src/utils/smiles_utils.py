# src/utils/smiles_utils.py
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional, Tuple
import pandas as pd
import os
from tqdm import tqdm

def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return canonical isomeric SMILES or None if cannot parse."""
    if not isinstance(smi, str) or smi.strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        c = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return c
    except Exception:
        return None

def batch_canonicalize(smiles: List[str], save_failed: Optional[str]=None) -> Tuple[List[Optional[str]], pd.DataFrame]:
    """Canonicalize a list of SMILES, optionally record failures to CSV."""
    results = []
    failed = []
    for i, s in enumerate(tqdm(smiles, desc="canonicalize")):
        cs = canonicalize_smiles(s)
        results.append(cs)
        if cs is None:
            failed.append({"index": i, "smiles": s})
    failed_df = pd.DataFrame(failed)
    if save_failed and not failed_df.empty:
        failed_df.to_csv(save_failed, index=False)
    return results, failed_df
