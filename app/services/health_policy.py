"""Runtime health-scoring policy shared by all calculation and API layers."""
from __future__ import annotations

import os


def baseline_score() -> float:
    """Read the configured baseline once per calculation, with safe bounds."""
    try:
        return max(0.0, min(100.0, float(os.environ.get("PHPS_HEALTH_BASELINE", "100.0"))))
    except ValueError:
        return 100.0
