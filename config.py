import torch

class CFG:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_folds = 5
    epochs = 80
    batch_size = 128

    lr = 1e-3
    weight_decay = 1e-5

    hidden_dim = 128
    num_layers = 3
    dropout = 0.2

    num_targets = 5
    fp_bits = 1024