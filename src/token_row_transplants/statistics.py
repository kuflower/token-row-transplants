"""Small-sample statistics used by the reported comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def _sample(values: Sequence[float]) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 2:
        raise ValueError("at least two paired estimates are required")
    if not bool(np.isfinite(sample).all()):
        raise ValueError("paired estimates must be finite")
    return sample


def mean_t_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return the sample mean and a two-sided Student t interval."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    sample = _sample(values)
    mean = float(sample.mean())
    standard_error = float(sample.std(ddof=1) / np.sqrt(len(sample)))
    half_width = float(
        stats.t.ppf(0.5 + confidence / 2, df=len(sample) - 1) * standard_error
    )
    return mean, mean - half_width, mean + half_width


def directional_p_value(
    values: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """Return a one-sided one-sample t-test p value for paired estimates."""

    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    result = stats.ttest_1samp(_sample(values), 0.0, alternative=alternative)
    return float(result.pvalue)


def equivalence_p_value(values: Sequence[float], *, margin: float) -> float:
    """Return the two one-sided tests p value for a symmetric range."""

    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    sample = _sample(values)
    lower_test = stats.ttest_1samp(sample, -margin, alternative="greater")
    upper_test = stats.ttest_1samp(sample, margin, alternative="less")
    return float(max(lower_test.pvalue, upper_test.pvalue))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down correction to one family of p values."""

    for name, value in p_values.items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p value for {name!r}")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_maximum = 0.0
    for rank, (name, value) in enumerate(ordered):
        running_maximum = max(running_maximum, (len(ordered) - rank) * value)
        adjusted[name] = min(1.0, running_maximum)
    return adjusted
