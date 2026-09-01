"""Pure trajectory metrics shared by grid and downstream analyses."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


class MetricError(ValueError):
    """Raised when a probe table cannot define complete trajectories."""


def trajectory_metrics(
    probe_path: Path,
    *,
    minimum_contexts: int,
    frame: pd.DataFrame | None = None,
    expected_steps: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Return raw step-zero/final surprisal and running-minimum log-time AUC.

    ``mean_log_probability`` is the summed target-token log probability recorded by
    the probe. This function deliberately leaves it on that raw scale; a
    caller that needs a per-piece estimand must form its paired contrast and
    apply the target-piece denominator explicitly.
    """
    probe = pd.read_csv(probe_path) if frame is None else frame.copy()
    required = {"mean_log_probability", "context_count", "english_updates", "word"}
    if not required.issubset(probe.columns):
        raise MetricError(f"probe columns missing in {probe_path}")
    if (
        isinstance(minimum_contexts, bool)
        or not isinstance(minimum_contexts, int)
        or minimum_contexts < 1
    ):
        raise MetricError("minimum_contexts must be positive")
    if probe.empty or bool(probe["word"].isna().any()):
        raise MetricError(f"probe word grid is empty or incomplete in {probe_path}")
    contexts = pd.to_numeric(probe["context_count"], errors="coerce")
    raw_steps = pd.to_numeric(probe["english_updates"], errors="coerce")
    log_probabilities = pd.to_numeric(probe["mean_log_probability"], errors="coerce")
    if (
        not bool(contexts.notna().all())
        or not bool(np.isfinite(contexts.to_numpy(dtype=float)).all())
        or bool((contexts < 1).any())
        or not bool(np.equal(contexts, np.floor(contexts)).all())
    ):
        raise MetricError(f"probe context counts are invalid in {probe_path}")
    if (
        not bool(raw_steps.notna().all())
        or not bool(np.isfinite(raw_steps.to_numpy(dtype=float)).all())
        or bool((raw_steps < 0).any())
        or not bool(np.equal(raw_steps, np.floor(raw_steps)).all())
    ):
        raise MetricError(f"probe steps are invalid in {probe_path}")
    if not bool(log_probabilities.notna().all()) or not bool(
        np.isfinite(log_probabilities.to_numpy(dtype=float)).all()
    ):
        raise MetricError(f"probe log probabilities are invalid in {probe_path}")
    probe["context_count"] = contexts.astype(int)
    probe["english_updates"] = raw_steps.astype(int)
    probe["mean_log_probability"] = log_probabilities.astype(float)
    maximum_contexts = probe.groupby("word")["context_count"].max()
    keep = set(maximum_contexts[maximum_contexts >= minimum_contexts].index)
    probe = probe[probe["word"].isin(keep)].copy()
    if probe.empty:
        raise MetricError(f"no words pass the context threshold in {probe_path}")
    probe["surprisal"] = -probe["mean_log_probability"]
    try:
        pivot = probe.pivot(
            index="word",
            columns="english_updates",
            values="surprisal",
        ).dropna()
    except ValueError as error:
        raise MetricError(
            f"probe has duplicate word/step rows in {probe_path}"
        ) from error
    if pivot.empty:
        raise MetricError(f"no complete word trajectories in {probe_path}")
    try:
        actual_steps = tuple(sorted(int(step) for step in pivot.columns))
    except (TypeError, ValueError) as error:
        raise MetricError(f"probe steps are invalid in {probe_path}") from error
    if len(actual_steps) != len(set(actual_steps)) or not actual_steps:
        raise MetricError(f"probe steps are ambiguous in {probe_path}")
    if actual_steps[0] != 0:
        raise MetricError(f"probe has no step-zero measurement in {probe_path}")
    if expected_steps is not None:
        if (
            not expected_steps
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step < 0
                for step in expected_steps
            )
            or expected_steps[0] != 0
            or any(left >= right for left, right in pairwise(expected_steps))
        ):
            raise MetricError("expected probe steps must increase from zero")
        if actual_steps != expected_steps:
            raise MetricError(
                f"probe step grid differs in {probe_path}: "
                f"{actual_steps} != {expected_steps}"
            )
    steps = np.asarray(actual_steps, dtype=float)
    values = pivot[list(actual_steps)].to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise MetricError(f"probe contains non-finite trajectories in {probe_path}")
    smoothed = np.minimum.accumulate(values, axis=1)
    terminal_best = smoothed[:, -1]
    learning_curve_auc = np.trapezoid(
        np.clip(smoothed - terminal_best[:, None], 0, None),
        np.log1p(steps),
        axis=1,
    )
    return pd.DataFrame(
        {
            "word": pivot.index.astype(str),
            "step_zero_surprisal": values[:, actual_steps.index(0)],
            "learning_curve_auc": learning_curve_auc,
            "final_surprisal": values[:, -1],
        }
    )
