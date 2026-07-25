"""Create reproducible snapshot data through the runtime feature engineer.

Production exports can replace the generated observations without changing the
feature matrix: both paths call ``build_feature_dict``.
"""
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app.ml.features import FEATURE_NAMES, build_feature_dict


def generate_accurate_dataset(snapshots=1500, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    previous_by_project = {}
    for index in range(snapshots):
        project = f"PROJ-{index % 50:03d}"
        state = rng.choice(("healthy", "normal", "risk"), p=(.2, .6, .2))
        if state == "healthy":
            tone, velocity, overdue, bug = rng.uniform(.35, .9), rng.uniform(.75, 1), rng.uniform(0, .1), rng.uniform(0, .1)
        elif state == "risk":
            tone, velocity, overdue, bug = rng.uniform(-.9, -.15), rng.uniform(.2, .55), rng.uniform(.25, .65), rng.uniform(.2, .55)
        else:
            tone, velocity, overdue, bug = rng.uniform(-.15, .45), rng.uniform(.5, .82), rng.uniform(.05, .28), rng.uniform(.05, .25)
        email_count, transcript_count = int(rng.integers(0, 15)), int(rng.integers(0, 6))
        open_issues = int(rng.integers(3, 45)); open_bugs = max(0, round(open_issues * bug))
        features = build_feature_dict(overall_sentiment=tone, urgency_count=int(rng.poisson(max(0, overdue * 5))),
            velocity=velocity, overdue_rate=overdue, bug_ratio=bug, open_issues=open_issues,
            open_bugs=open_bugs, email_count=email_count, transcript_count=transcript_count,
            deadline=datetime.now() + timedelta(days=int(rng.integers(7, 240))), previous=previous_by_project.get(project))
        # Label generation is isolated to research data creation; the model
        # never receives this formula at production inference time.
        health = np.clip(52 + 24 * features["velocity"] - 18 * features["overdue_rate"] - 14 * features["bug_ratio"] + 19 * features["overall_sentiment"] - 2 * features["urgency_count"] + rng.normal(0, 3), 0, 100)
        rows.append({**features, "health_score": round(float(health), 2)})
        previous_by_project[project] = features
    output = Path(__file__).resolve().parent / "accurate_fusion_dataset.csv"
    pd.DataFrame(rows, columns=[*FEATURE_NAMES, "health_score"]).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    print(generate_accurate_dataset())
