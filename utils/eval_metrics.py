# utils/eval_metrics.py

import numpy as np
from fastdtw import fastdtw
from scipy.stats import pearsonr

def compute_dtw(true_window, pred_window):
    """
    Compute the DTW distance between two 1D signals.
    """
    true_window = np.ravel(true_window)
    pred_window = np.ravel(pred_window)
    distance, _ = fastdtw(true_window.tolist(), pred_window.tolist())
    return distance

def compute_pearson(true_window, pred_window):
    """
    Compute the Pearson correlation coefficient between two 1D signals.
    """
    true_window = np.ravel(true_window)
    pred_window = np.ravel(pred_window)
    corr, _ = pearsonr(true_window, pred_window)
    return corr

def compute_rmse(true_window, pred_window):
    """
    Compute the RMSE between two 1D signals.
    """
    true_window = np.ravel(true_window)
    pred_window = np.ravel(pred_window)
    return np.sqrt(np.mean((true_window - pred_window) ** 2))
