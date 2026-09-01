from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from token_row_transplants.probe import EncodedProbe, target_piece_counts


class UniformModel:
    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.training = True

    def eval(self) -> UniformModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> UniformModel:
        self.training = mode
        return self

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        shape = (*token_ids.shape, self.vocabulary_size)
        return torch.zeros(shape, dtype=torch.float32)


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "plain": 1, "context": 2, "alpha": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_probe_encodes_contexts_and_scores_whole_words(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    probe_path = tmp_path / "probe.jsonl"
    _write_tokenizer(tokenizer_path)
    rows = [
        {"prefix": "plain context", "word": "alpha"},
        {"prefix": "context", "word": "alpha"},
    ]
    probe_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    probe = EncodedProbe.from_jsonl(
        probe_path=probe_path,
        tokenizer_path=tokenizer_path,
        vocabulary_size=4,
        context_length=4,
    )
    scores = probe.score(UniformModel(4), device="cpu", batch_size=2)

    assert probe.context_counts() == {"alpha": 2}
    assert scores.values[0][0] == "alpha"
    assert scores.values[0][1] == pytest.approx(-math.log(4))
    assert scores.values[0][2] == 2
    assert target_piece_counts(
        ["alpha"], tokenizer_path=tokenizer_path, vocabulary_size=4
    ) == {"alpha": 1}
