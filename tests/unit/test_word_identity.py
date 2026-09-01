from __future__ import annotations

import pytest

from token_row_transplants.word_identity import (
    FeatureContractError,
    normalize_word,
    normalized_context_counts,
)


def test_word_normalization_uses_unicode_case_folding() -> None:
    assert normalize_word("  Straße ") == "strasse"


def test_normalized_context_counts_reject_colliding_spellings() -> None:
    with pytest.raises(FeatureContractError, match="collide"):
        normalized_context_counts({"Straße": 2, "STRASSE": 3})
