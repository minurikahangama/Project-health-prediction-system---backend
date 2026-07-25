"""Constrained one-week health optimisation; no arbitrary target overrides."""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from app.ml.preprocessor import clip_health

OPTIMIZED = ("overall_sentiment", "velocity", "overdue_rate", "bug_ratio")

def optimize_health(features: dict, predict: Callable[[dict], float]) -> dict:
    baseline = {key: float(features.get(key, 0.0)) for key in features}
    x0 = np.array([baseline.get(key, 0.0) for key in OPTIMIZED], dtype=float)
    bounds = [(-1, 1), (0, 1), (0, 1), (0, 1)]

    def candidate(x):
        result = {**baseline, **dict(zip(OPTIMIZED, x))}
        # Fewer defects cannot leave the historic issue count unchanged.
        if baseline.get("bug_ratio", 0) > 0 and x[3] < baseline["bug_ratio"]:
            result["open_bugs"] = max(0.0, baseline.get("open_bugs", 0) * x[3] / baseline["bug_ratio"])
        return result

    constraints = [
        {"type": "ineq", "fun": lambda x: .15 - abs(x[1] - x0[1])},
        {"type": "ineq", "fun": lambda x: .15 - abs(x[2] - x0[2])},
        {"type": "ineq", "fun": lambda x: .10 - abs(x[3] - x0[3])},
        {"type": "ineq", "fun": lambda x: .30 - abs(x[0] - x0[0])},
        # A velocity increase consumes capacity and may increase defect risk slightly.
        {"type": "ineq", "fun": lambda x: .03 + (x[3] - x0[3]) + .20 * (x[1] - x0[1])},
    ]
    result = minimize(lambda x: -clip_health(predict(candidate(x))), x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 80, "ftol": .01})
    best = candidate(result.x if result.success else x0)
    current, predicted = clip_health(predict(baseline)), clip_health(predict(best))
    adjustments = {key: round(best[key] - baseline[key], 4) for key in OPTIMIZED}
    return {"current_score": current, "optimized_score": predicted, "estimated_gain": round(predicted-current, 2),
            "optimized_features": best, "adjustments": adjustments,
            "solver_status": result.message, "solver_success": bool(result.success)}


def recovery_plan(optimization: dict) -> list[dict]:
    changes = optimization["adjustments"]
    phases = [
        ("Week 1", "Stabilise defects and communication", ["bug_ratio", "overall_sentiment"]),
        ("Week 2", "Remove overdue work, then increase delivery throughput", ["overdue_rate", "velocity"]),
        ("Week 3", "Verify the health improvement and maintain controls", []),
    ]
    return [{"week": week, "phase": phase, "metrics": {key: changes[key] for key in keys},
             "expected_score": optimization["optimized_score"] if week == "Week 3" else None}
            for week, phase, keys in phases]
