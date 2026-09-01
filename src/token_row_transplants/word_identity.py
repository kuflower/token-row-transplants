"""Word normalization shared by probe and analysis code."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping


class FeatureContractError(ValueError):
    """Raised when feature identity or required values are ambiguous."""


def normalize_word(value: str) -> str:
    """Normalize a word with NFKC, trimming, and Unicode case folding."""
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if not normalized:
        raise FeatureContractError("word normalization produced an empty value")
    return normalized


def normalized_context_counts(
    context_counts: Mapping[str, int],
) -> dict[str, int]:
    """Normalize probe identities and reject spelling collisions."""
    normalized: dict[str, int] = {}
    spellings: dict[str, set[str]] = {}
    for raw_word, count in context_counts.items():
        if not isinstance(raw_word, str) or not isinstance(count, int) or count < 1:
            raise FeatureContractError("probe context counts are invalid")
        word = normalize_word(raw_word)
        normalized[word] = normalized.get(word, 0) + count
        spellings.setdefault(word, set()).add(raw_word)
    collisions = {
        word: sorted(values) for word, values in spellings.items() if len(values) != 1
    }
    if collisions:
        raise FeatureContractError(
            f"probe spellings collide after normalization: {collisions}"
        )
    return normalized
