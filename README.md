# Kaggle-NeurIPS---Open-Polymer-Prediction-2025-Silver-Algorithm-Overview
1.Competition Overview<br>
1.1 Background<br>
  Filter the R-group and suspicious patterns in the polymer notation, use RDKit for parsability verification and unified conversion on this basis, add multi-source external data, grouping and mean aggregation with cS as the key to de-weight, multi-view GNN, CatBoost, XGboost modelling, using linear weighting for model fusion.<br>
  As a solo participant, I independently developed a complete solution for the Kaggle NeurIPS 2025-Open Polymer Prediction challenge. Organized by the Neural Information Processing Systems Conference (NeurIPS), this competition addresses critical issues in polymer materials science: the accurate prediction of physical and chemical properties of SMILES-based polymer strings.<br>
1.2 Tasks and Challenges<br>
Task: Direct prediction of five physical properties (Tg/FFV/Tc/Density/Rg) from polymer SMILES. Using approximately 8,000 polymer SMILES data lines, the weighted mean absolute error (wMAE) was employed as the evaluation metric.<br>
Data heterogeneity: The SMILES of polymers contain R-group markers and multiple isomeric representations, resulting in inconsistent input features, and the labels are derived from the mean of molecular dynamics simulations, inherently containing noise.<br>
Multitasking complexity: requires simultaneous optimization of five regression tasks with different dimensions and physical meanings.<br>
Computing resource limits: Kaggle Notebook environment restrictions (no internet access, single run ≤9 hours)<br>
2. The algorithm used<br>
As an individual participant, I independently designed and implemented an end-to-end solution integrating data governance, multi-perspective representation, and weighted fusion. The process enhances generalization capabilities through model diversity, as detailed below:<br>
2.1 Expansion of Data Governance and External Supervision<br>
Filter R-groups and suspicious patterns in polymer tags, perform parseability verification using RDKit, and convert them uniformly into canonical SMILES to ensure consistent representation of isomeric molecules at the input level. On this basis, incorporate multi-source Tg/Tc/Density/FFV external data, grouping and averaging with canonical SMILES as the key to deduplicate, as illustrated.<br>
2.2 Multi-perspective<br>
Representation, Feature Engineering and Model Training<br>
a) Molecular Graph Representation (GNN Approach): <br>The molecule is decomposed into a graph structure composed of nodes and edges. Node features include atomic number, degree, formal charge, hybridization state, aromaticity, total hydrogen count, ring presence, and atomic mass. Edge features encompass bond type, ring presence, conjugation status, and aromaticity, with undirected graph representation. Global molecular features (MolWt, HBD, HBA, TPSA, rotatable bond count, and SMILES length) are additionally incorporated to enhance graph-level representation. The model extracts local structures through GCN stacking, followed by GAT attention layers to strengthen information aggregation. Node representations undergo mean and max pooling before being concatenated with molecular-level features. An independent regression head is configured for each target at the output end. <br>
b) Chemical descriptor representation (CatBoost approach): <br>The Mordred tool calculates large-scale 2D descriptors. The training side processes precomputed feature tables by removing constant and non-numeric columns, while the testing side performs real-time computations on the test set and aligns with the training set columns. Regression analysis is conducted using the CatBoost model based on Mordred descriptors.<br>
c) Statistical representation of fingerprints and images (XGBoost approach): <br>Constructing Morgan fingerprints (2128-bit radius) and MACCS 166-bit fingerprints:
Features: Morgan (r=2,128 bits) + MACCS (166) + RDKit physicalization + NetworkX graph statistics (diameter/shortest path/circumference)
Extract RDKit materialization descriptors and NetworkX-based graph statistics (e.g., graph diameter, average shortest path, and number of cycles). The final input is fed into the XGBoost model for training. 
2.3 Model Fusion
The model fusion employs linear weighting with a configuration of 0.4 GNN, 0.3 CatBoost, and 0.3 XGBoost, where each target is independently weighted. 
