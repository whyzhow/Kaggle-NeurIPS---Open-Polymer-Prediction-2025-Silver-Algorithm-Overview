# Kaggle NeurIPS – Open Polymer Prediction  
### Reproducible and Improved Silver-Medal Solution

This repository is an engineering-refined and fully reproducible implementation based on  
**Kaggle NeurIPS 2025 – Open Polymer Prediction (Silver Medal)** solution.

The pipeline combines:
- SMILES canonicalization and data cleaning (RDKit)
- Fingerprint + descriptor feature engineering
- Group-aware cross-validation (to avoid data leakage)
- Traditional ML models (CatBoost, XGBoost)
- Optional GNN model (PyTorch Geometric)
- Out-of-Fold (OOF) stacking and ensemble learning
- Optuna-based weight optimization (optional)

The goal is to provide:<br>
✅ higher stability  
✅ better generalization  
✅ strict reproducibility  
✅ clean project structure  
## 📂 Project Structure
Kaggle-NeurIPS-Open-Polymer/<br>
├── README.md<br>
├── requirements.txt<br>
├── config.py<br>
├── dataset.csv<br>            
├── src/<br>
│   ├── train.py<br>
│   ├── features.py<br>
│   ├── models.py<br>
│   ├── utils/<br>
│   │   ├── smiles_utils.py<br>
│   │   └── cv_utils.py<br>
│   └── stacking/<br>
│       └── optuna_stack.py<br>
├── outputs/<br>
│   ├── models/<br>
│   ├── feats/<br>
│   └── oof/<br>
└── tests/<br>
└── test_smiles.py<br>
## ⚙️ Environment Setup (Recommended)

We strongly recommend using **Conda** to install RDKit.

### 1. Create environment
```
conda create -n polymer_env python=3.10 -y
conda activate polymer_env
```
### 2. Install RDKit
```
conda install -c conda-forge rdkit -y
```
### 3. Install remaining dependencies
```
pip install -r requirements.txt
```
## 🔬 Pipeline Overview
### 1.	SMILES Canonicalization
	•	Uses RDKit to convert SMILES into canonical isomeric form
	•	Invalid SMILES are logged to outputs/oof/failed_smiles.csv
### 	2.	Feature Engineering
	•	Morgan fingerprints (2048 bits)
	•	Truncated SVD for dimensionality reduction
	•	RDKit descriptors (MolWt, TPSA, LogP)
### 	3.	Cross Validation
	•	GroupKFold based on canonical SMILES
	•	Prevents molecule-level leakage
### 4.	Models
	•	CatBoostRegressor
	•	XGBoostRegressor
	•	Optional GNN (PyTorch Geometric)
### 5.	Ensemble
	•	Ridge regression as meta-learner
	•	Optional Optuna-optimized weighted ensemble
### 6.	Reproducibility
	•	Fixed random seed
	•	Cached features
	•	Saved models and OOF predictions
## 🚀 Quick Start (Demo Mode)

Run a small synthetic dataset to verify installation:
```
python src/train.py --demo
```
This will:
	•	Generate a fake dataset<br>
	•	Train models<br>
	•	Output results to outputs/<br>
## 🧩 Extending the Project

You can easily add:
	•	New molecular descriptors<br>
	•	New GNN architectures (GAT, GraphSAGE)<br>
	•	New meta learners (LightGBM, ElasticNet)<br>
	•	Bayesian uncertainty estimation<br>
	•	Multi-task prediction heads<br>
