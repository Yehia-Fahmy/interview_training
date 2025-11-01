from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e_hist, edges = np.histogram(expected, bins=bins, density=True)
    a_hist, _ = np.histogram(actual, bins=edges, density=True)
    e_hist = np.clip(e_hist, 1e-8, None)
    a_hist = np.clip(a_hist, 1e-8, None)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    stat, _ = stats.ks_2samp(a, b)
    return float(stat)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        conf = np.mean(probs[mask])
        acc = np.mean(labels[mask])
        ece += (np.sum(mask) / len(probs)) * abs(acc - conf)
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    confs, accs = [], []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            confs.append((lo + hi) / 2)
            accs.append(np.nan)
            continue
        confs.append(np.mean(probs[mask]))
        accs.append(np.mean(labels[mask]))
    return np.array(confs), np.array(accs)


