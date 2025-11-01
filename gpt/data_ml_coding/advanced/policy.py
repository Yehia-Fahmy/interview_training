"""Decision policy scaffolding for automated agent governance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class Thresholds:
    min_success_rate: float = 0.92
    max_safety_violation_rate: float = 0.01
    max_disagreement_rate: float = 0.08
    max_avg_latency_ms: float = 2_000.0


def decide_action(metrics: Mapping[str, float], thresholds: Thresholds) -> str:
    """Return one of {continue, rollback, trigger_retraining}."""

    success = metrics.get("success_rate", 0.0)
    safety = metrics.get("safety_violation_rate", 1.0)
    disagreement = metrics.get("disagreement_rate", 1.0)
    latency = metrics.get("avg_latency_ms", thresholds.max_avg_latency_ms)

    if safety > thresholds.max_safety_violation_rate:
        return "rollback"
    if success < thresholds.min_success_rate:
        return "trigger_retraining"
    if disagreement > thresholds.max_disagreement_rate:
        return "trigger_retraining"
    if latency > thresholds.max_avg_latency_ms:
        return "trigger_retraining"
    return "continue"

