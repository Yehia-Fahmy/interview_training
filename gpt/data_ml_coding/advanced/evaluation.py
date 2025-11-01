"""Metric calculators for the advanced agent evaluation exercise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping


@dataclass
class Interaction:
    ticket_id: str
    severity: str
    outcome: bool
    safety_violation: bool
    latency_ms: float
    drafter: str
    reviewer: str
    disagreement: bool


def load_interactions(records: Iterable[Mapping[str, object]]) -> List[Interaction]:
    interactions: List[Interaction] = []
    for record in records:
        interactions.append(
            Interaction(
                ticket_id=str(record["ticket_id"]),
                severity=str(record.get("severity", "normal")),
                outcome=bool(record.get("outcome", False)),
                safety_violation=bool(record.get("safety_violation", False)),
                latency_ms=float(record.get("latency_ms", 0.0)),
                drafter=str(record.get("drafter", "agent@draft")),
                reviewer=str(record.get("reviewer", "agent@review")),
                disagreement=bool(record.get("disagreement", False)),
            )
        )
    return interactions


def severity_weight(severity: str) -> float:
    return {
        "critical": 2.0,
        "high": 1.5,
        "normal": 1.0,
        "low": 0.5,
    }.get(severity, 1.0)


def compute_metrics(interactions: List[Interaction]) -> MutableMapping[str, float]:
    total_weight = 0.0
    success_weight = 0.0
    safety_weight = 0.0
    total_latency = 0.0
    disagreements = 0

    for interaction in interactions:
        weight = severity_weight(interaction.severity)
        total_weight += weight
        if interaction.outcome:
            success_weight += weight
        if interaction.safety_violation:
            safety_weight += weight
        total_latency += interaction.latency_ms
        if interaction.disagreement:
            disagreements += 1

    if not interactions:
        return {
            "success_rate": 0.0,
            "safety_violation_rate": 0.0,
            "avg_latency_ms": 0.0,
            "disagreement_rate": 0.0,
        }

    count = len(interactions)
    return {
        "success_rate": success_weight / total_weight if total_weight else 0.0,
        "safety_violation_rate": safety_weight / total_weight if total_weight else 0.0,
        "avg_latency_ms": total_latency / count,
        "disagreement_rate": disagreements / count,
    }

