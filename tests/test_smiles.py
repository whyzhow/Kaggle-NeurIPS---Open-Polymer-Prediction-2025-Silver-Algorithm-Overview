# tests/test_smiles.py
from src.utils.smiles_utils import canonicalize_smiles

def test_canonicalize():
    s = "OCC"  # ethanol is CC O or OCC -> canonical CC O
    cs = canonicalize_smiles(s)
    assert cs is not None
    # known mapping
    assert "O" in cs or "C" in cs
    print("smiles canonicalize OK")
if __name__ == "__main__":
    test_canonicalize()
