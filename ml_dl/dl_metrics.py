import torch, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, brier_score_loss

def auc(y_true, y_prob):
    y = y_true.detach().cpu().numpy()
    p = y_prob.detach().cpu().numpy()
    mask = ~np.isnan(y) & ~np.isnan(p)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2: return np.nan
    return roc_auc_score(y[mask], p[mask])

def mse_mae(y_true, y_pred):
    y = y_true.detach().cpu().numpy()
    p = y_pred.detach().cpu().numpy()
    m = ~np.isnan(y) & ~np.isnan(p)
    if m.sum() < 2: return np.nan, np.nan
    return mean_squared_error(y[m], p[m]), mean_absolute_error(y[m], p[m])

def information_coefficient(y_true_ret, y_pred_score):
    # Spearman rank correlation between predicted score and future returns
    y = y_true_ret.detach().cpu().numpy()
    s = y_pred_score.detach().cpu().numpy()
    m = ~np.isnan(y) & ~np.isnan(s)
    if m.sum() < 3: return np.nan
    return spearmanr(s[m], y[m]).correlation

def calibration_ece(y_true, y_prob, n_bins=10):
    # simple ECE; for binary classification
    y = y_true.detach().cpu().numpy()
    p = y_prob.detach().cpu().numpy()
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) < 10: return np.nan
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        idx = (p >= bins[i]) & (p < bins[i+1])
        if idx.sum() == 0: continue
        ece += np.abs(p[idx].mean() - y[idx].mean()) * (idx.sum()/len(y))
    return ece
