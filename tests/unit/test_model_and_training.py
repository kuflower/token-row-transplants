from __future__ import annotations

import numpy as np
import torch

from token_row_transplants.model import TiedDecoder
from token_row_transplants.training import (
    TrainingSettings,
    sample_token_batch,
    train_language_model,
)


def test_tied_decoder_uses_one_token_matrix() -> None:
    model = TiedDecoder(
        vocabulary_size=16,
        context_length=4,
        width=8,
        layers=1,
        heads=2,
    )

    assert model.head.weight is model.tok.weight
    assert model(torch.tensor([[1, 2, 3, 4]])).shape == (1, 4, 16)


def test_batch_sampling_and_short_training_run() -> None:
    tokens = np.tile(np.arange(8, dtype=np.int64), 8)
    inputs, targets = sample_token_batch(
        tokens,
        batch_size=3,
        context_length=4,
        generator=np.random.default_rng(5),
    )
    assert inputs.shape == targets.shape == (3, 4)
    assert np.array_equal(inputs[:, 1:], targets[:, :-1])

    model = TiedDecoder(
        vocabulary_size=8,
        context_length=4,
        width=8,
        layers=1,
        heads=2,
    )
    losses = train_language_model(
        model,
        tokens,
        TrainingSettings(updates=2, batch_size=2, warmup_updates=1),
        seed=7,
    )
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)
