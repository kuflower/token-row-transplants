from __future__ import annotations

import pytest

from token_row_transplants.assignment import (
    frequency_local_reassignment,
    validate_reassignment,
)


def test_frequency_local_reassignment_is_reproducible_and_complete() -> None:
    frequencies = [20, 18, 14, 9, 7, 4, 3, 1]

    first = frequency_local_reassignment(
        frequencies,
        seed=19301,
        maximum_block_size=4,
    )
    second = frequency_local_reassignment(
        frequencies,
        seed=19301,
        maximum_block_size=4,
    )

    assert first == second
    assert set(first) == set(range(len(frequencies)))
    assert all(token_id != row_id for token_id, row_id in enumerate(first))


def test_validate_reassignment_rejects_fixed_points() -> None:
    with pytest.raises(ValueError, match="fixed points"):
        validate_reassignment([0, 2, 1])
