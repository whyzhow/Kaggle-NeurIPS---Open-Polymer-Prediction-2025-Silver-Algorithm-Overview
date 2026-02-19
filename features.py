import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator


def smiles_to_combined_features(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    morgan_fp = morgan_gen.GetFingerprint(mol)

    # AtomPair
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    atom_pair_fp = atom_pair_gen.GetFingerprint(mol)

    # MACCS
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)

    combined_fp = np.concatenate([
        np.array(morgan_fp),
        np.array(atom_pair_fp),
        np.array(maccs_fp)
    ])

    # RDKit Descriptors
    desc_values = []
    for name, func in Descriptors.descList:
        try:
            desc_values.append(func(mol))
        except:
            desc_values.append(0)

    desc_values = np.array(desc_values)

    return np.concatenate([combined_fp, desc_values])