from __future__ import annotations

import math
import re
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    s = text.lower()
    s = _PUNCT.sub("", s)
    s = _WS.sub(" ", s)
    return s.strip()


def exact_match(pred: str, ref: str) -> int:
    return int(normalize_text(pred) == normalize_text(ref))


def f1_score(pred: str, ref: str) -> float:
    p_tokens = normalize_text(pred).split()
    r_tokens = normalize_text(ref).split()
    common = {}
    for t in p_tokens:
        if t in r_tokens:
            common[t] = min(p_tokens.count(t), r_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(p_tokens), 1)
    recall = num_same / max(len(r_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (norm(vec_a) * norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def rouge_lite(pred: str, ref: str) -> Tuple[float, float]:
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    overlap = len(set(p) & set(r))
    prec = overlap / max(len(p), 1)
    rec = overlap / max(len(r), 1)
    return prec, rec


