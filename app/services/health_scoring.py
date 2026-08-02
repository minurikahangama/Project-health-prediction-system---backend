"""Pure, configuration-driven project health scoring calculations.

This module deliberately has no database or framework dependency.  It is the
single source of truth for the 100-point deduction model and is safe to unit
test independently from Jira and RoBERTa adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from typing import Iterable, Mapping

DECAY_LAMBDA = 0.05


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def status_for_score(score: float, green_threshold: float, amber_threshold: float) -> str:
    if not (0 <= amber_threshold < green_threshold <= 100):
        raise ValueError("Amber threshold must be lower than green threshold and both must be within 0-100")
    return "GREEN" if score >= green_threshold else "AMBER" if score >= amber_threshold else "RED"


def delivery_deductions(metrics: Mapping[str, object] | None) -> dict[str, float]:
    """Calculate delivery deductions from active-sprint Jira evidence only."""
    if not metrics:
        return {"velocity": 0.0, "overdue": 0.0, "bug": 0.0, "total": 0.0}
    active = bool(metrics.get("is_sprint_active"))
    committed = max(0.0, float(metrics.get("committed_story_pts") or 0))
    completed = max(0.0, float(metrics.get("actual_completed_story_pts", metrics.get("completed_story_pts", 0)) or 0))
    expected = clamp(float(metrics.get("expected_velocity_ratio") or 0), 0, 1)
    actual = clamp(completed / committed, 0, 1) if committed else 0.0
    velocity = 0.0
    # A sprint needs enough elapsed time before progress is meaningful.  Once
    # it is, allow a 10% progress tolerance before applying a proportional
    # velocity deduction.
    if active and committed > 0 and expected > .10 and actual < expected * .90:
        velocity = min(35.0, 35.0 * (1.0 - actual / expected))

    tickets = max(0, int(metrics.get("active_sprint_issue_count") or 0))
    overdue_count = max(0, int(metrics.get("overdue_issue_count") or 0))
    critical_count = max(0, int(metrics.get("critical_bug_count") or 0)) + max(0, int(metrics.get("blocker_bug_count") or 0))
    # `open_bug_count` is retained only as a compatibility fallback for older
    # snapshots that were created before severity counts existed.
    if not critical_count and "critical_bug_count" not in metrics and "blocker_bug_count" not in metrics:
        critical_count = max(0, int(metrics.get("open_bug_count") or 0))
    overdue = min(20.0, 20.0 * ((overdue_count / tickets) / .20)) if tickets else 0.0
    bug = min(25.0, 25.0 * ((critical_count / tickets) / .15)) if tickets else 0.0
    # The delivery portion of the top-down model is deliberately bounded,
    # even when several independent Jira risks are present at once.
    return {"velocity": velocity, "overdue": overdue, "bug": bug,
            "total": min(50.0, velocity + overdue + bug)}


def decay_weight(timestamp: datetime | None, now: datetime | None = None) -> float:
    if timestamp is None:
        return 1.0
    now = now or datetime.now(timezone.utc)
    timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
    return exp(-DECAY_LAMBDA * age_days)


def communication_deduction(items: Iterable[object], now: datetime | None = None) -> dict[str, float]:
    penalty = recovery = 0.0
    for item in items:
        tone = clamp(float(getattr(item, "tone_score", 0) or 0), -1, 1)
        urgency = 1.5 if int(bool(getattr(item, "urgency_flag", 0))) else 1.0
        timestamp = getattr(item, "processed_at", None) or getattr(item, "uploaded_at", None)
        weight = decay_weight(timestamp, now)
        item_penalty = abs(tone) * urgency * 15 * weight if tone < -.20 else 0.0
        item_recovery = tone * 8 * weight if tone > .20 else 0.0
        # Persistable derived values are set when model instances are passed.
        for name, value in (("penalty", item_penalty), ("recovery", item_recovery), ("decay_weight", weight)):
            if hasattr(item, name):
                setattr(item, name, value)
        penalty += item_penalty
        recovery += item_recovery
    net = penalty - recovery
    return {"penalty": penalty, "recovery": recovery, "net": net, "total": clamp(net, 0, 20)}


def health_score(metrics: Mapping[str, object] | None, communications: Iterable[object], green_threshold: float, amber_threshold: float) -> dict[str, object]:
    communications = tuple(communications)
    delivery = delivery_deductions(metrics)
    communication = communication_deduction(communications)
    score = clamp(100.0 - delivery["total"] - communication["total"], 0, 100)
    tone = sum(clamp(float(getattr(item, "tone_score", 0) or 0), -1, 1) for item in communications) / len(communications) if communications else 0.0
    delivery_score = 100.0 - delivery["total"]
    if delivery_score >= 80 and tone <= -0.30:
        divergence_key, divergence_label = "TEAM_BURNOUT_RISK", "Team Burnout Risk"
    elif delivery_score <= 50 and tone >= 0.50:
        divergence_key, divergence_label = "UNREPORTED_BLOCKERS_SCOPE_CREEP", "Unreported Blockers / Scope Creep"
    else:
        divergence_key, divergence_label = "NONE_DETECTED", "None Detected"
    return {"health_score": round(score, 2), "rag_status": status_for_score(score, green_threshold, amber_threshold), "delivery": delivery, "communication": communication, "divergence_flag": int(divergence_key != "NONE_DETECTED"), "divergence_key": divergence_key, "divergence_label": divergence_label}
