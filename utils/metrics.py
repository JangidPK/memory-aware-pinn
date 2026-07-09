"""utils/metrics.py -- simple error metrics for comparing prediction vs truth."""

import numpy as np


def mse(pred, true):
    return float(np.mean((pred - true) ** 2))


def relative_l2_error(pred, true):
    """||pred - true||_2 / ||true||_2"""
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true) + 1e-12
    return float(num / den)
