from __future__ import annotations

from pathlib import Path

import pandas as pd

from token_row_transplants.metrics import trajectory_metrics


def test_trajectory_metrics_use_the_running_minimum(tmp_path: Path) -> None:
    rows = []
    for word, values in {"alpha": [4.0, 3.0, 3.5], "beta": [2.0, 1.5, 1.0]}.items():
        for step, surprisal in zip((0, 10, 100), values, strict=True):
            rows.append(
                {
                    "word": word,
                    "english_updates": step,
                    "mean_log_probability": -surprisal,
                    "context_count": 5,
                }
            )
    frame = pd.DataFrame(rows)

    result = trajectory_metrics(
        tmp_path / "unused.csv",
        minimum_contexts=5,
        frame=frame,
        expected_steps=(0, 10, 100),
    )

    assert result["word"].tolist() == ["alpha", "beta"]
    assert result["step_zero_surprisal"].tolist() == [4.0, 2.0]
    assert result["final_surprisal"].tolist() == [3.5, 1.0]
    assert bool((result["learning_curve_auc"] > 0).all())
