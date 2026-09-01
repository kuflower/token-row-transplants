from __future__ import annotations

import pytest

from token_row_transplants.statistics import (
    directional_p_value,
    equivalence_p_value,
    holm_adjust,
    mean_t_interval,
)


def test_interval_and_directional_test_use_paired_estimates() -> None:
    values = [0.18, 0.23, 0.19, 0.25, 0.21, 0.20]

    mean, low, high = mean_t_interval(values)

    assert mean == pytest.approx(0.21)
    assert 0 < low < mean < high
    assert directional_p_value(values) < 0.001


def test_equivalence_and_holm_adjustment() -> None:
    values = [-0.01, 0.00, 0.01, 0.00, -0.02, 0.02]

    assert equivalence_p_value(values, margin=0.10) < 0.01
    assert holm_adjust({"small": 0.01, "large": 0.20}) == {
        "small": 0.02,
        "large": 0.20,
    }
