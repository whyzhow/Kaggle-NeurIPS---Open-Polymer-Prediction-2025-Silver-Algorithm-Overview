# src/stacking/optuna_stack.py
import optuna
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def optimize_weights(oof_preds, y_true, n_trials=100, seed=42):
    """
    oof_preds: np.array shape (n_models, n_samples)
    y_true: (n_samples,)
    Returns best weights normalized.
    """
    n_models = oof_preds.shape[0]
    def objective(trial):
        ws = []
        for i in range(n_models):
            ws.append(trial.suggest_float(f"w{i}", 0.0, 1.0))
        ws = np.array(ws)
        if ws.sum() <= 0:
            return 1e9
        ws = ws / ws.sum()
        pred = (ws.reshape(-1,1) * oof_preds).sum(axis=0)
        return mean_absolute_error(y_true, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    w = np.array([best[f"w{i}"] for i in range(n_models)])
    return w / w.sum()
