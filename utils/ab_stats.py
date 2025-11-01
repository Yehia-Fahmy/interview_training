from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def two_sample_ttest(a: np.ndarray, b: np.ndarray, equal_var: bool = False) -> Tuple[float, float]:
    t, p = stats.ttest_ind(a, b, equal_var=equal_var)
    return float(t), float(p)


def bootstrap_ci(x: np.ndarray, stat_fn, n_boot: int = 2000, alpha: float = 0.05, rng: int = 42) -> Tuple[float, float]:
    rnd = np.random.default_rng(rng)
    stats_list = []
    n = len(x)
    for _ in range(n_boot):
        sample = x[rnd.integers(0, n, size=n)]
        stats_list.append(stat_fn(sample))
    lower = np.quantile(stats_list, alpha / 2)
    upper = np.quantile(stats_list, 1 - alpha / 2)
    return float(lower), float(upper)


