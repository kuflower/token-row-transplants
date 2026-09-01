"""Row-to-token reassignments used by the assignment experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence


def validate_reassignment(source_row_for_token: Sequence[int]) -> tuple[int, ...]:
    """Return a checked permutation with no token left on its original row."""

    assignment = tuple(source_row_for_token)
    if len(assignment) < 2:
        raise ValueError("a reassignment needs at least two token rows")
    if any(isinstance(row, bool) or not isinstance(row, int) for row in assignment):
        raise TypeError("row indices must be integers")
    if set(assignment) != set(range(len(assignment))):
        raise ValueError("row indices must form a permutation")
    if any(token_id == row_id for token_id, row_id in enumerate(assignment)):
        raise ValueError("a reassignment cannot contain fixed points")
    return assignment


def frequency_local_reassignment(
    token_frequencies: Sequence[float],
    *,
    seed: int,
    maximum_block_size: int = 33,
) -> tuple[int, ...]:
    """Reassign rows within neighboring frequency ranks.

    The return value is indexed by token ID. Each value gives the donor-row ID
    installed for that token. Every row is used exactly once.
    """

    frequencies = tuple(float(value) for value in token_frequencies)
    if len(frequencies) < 2:
        raise ValueError("at least two token frequencies are required")
    if any(value < 0 or value != value for value in frequencies):
        raise ValueError("token frequencies must be finite and nonnegative")
    if maximum_block_size < 2:
        raise ValueError("maximum_block_size must be at least two")

    ranked_tokens = sorted(
        range(len(frequencies)), key=lambda token_id: (-frequencies[token_id], token_id)
    )
    block_size = min(maximum_block_size, len(ranked_tokens))
    blocks = [
        ranked_tokens[start : start + block_size]
        for start in range(0, len(ranked_tokens), block_size)
    ]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-1].insert(0, blocks[-2].pop())

    generator = random.Random(seed)
    source_row_for_token = [-1] * len(frequencies)
    for block in blocks:
        generator.shuffle(block)
        for source_row, target_token in zip(block, block[1:] + block[:1], strict=True):
            source_row_for_token[target_token] = source_row
    return validate_reassignment(source_row_for_token)
