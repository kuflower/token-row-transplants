"""Language-model training used for parent runs and English continuation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as torch_functional

from .model import TiedDecoder, cosine_learning_rate


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    updates: int
    batch_size: int = 64
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_updates: int = 200
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if not 0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("learning-rate bounds are invalid")
        if not 0 <= self.warmup_updates <= self.updates:
            raise ValueError("warmup_updates must lie within the run")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


def sample_token_batch(
    token_ids: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample contiguous next-token examples from a one-dimensional stream."""

    tokens = np.asarray(token_ids)
    if tokens.ndim != 1 or len(tokens) <= context_length:
        raise ValueError("token stream is too short for the requested context")
    if batch_size < 1 or context_length < 2:
        raise ValueError("batch_size and context_length are invalid")
    starts = generator.integers(
        0,
        len(tokens) - context_length,
        size=batch_size,
    )
    offsets = np.arange(context_length)
    inputs = tokens[starts[:, None] + offsets]
    targets = tokens[starts[:, None] + offsets + 1]
    return inputs.astype(np.int64), targets.astype(np.int64)


def train_language_model(
    model: TiedDecoder,
    token_ids: np.ndarray,
    settings: TrainingSettings,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    token_id_offset: int = 0,
    on_update: Callable[[int, float], None] | None = None,
) -> tuple[float, ...]:
    """Train a tied decoder and return one loss value per update."""

    if seed < 0 or token_id_offset < 0:
        raise ValueError("seed and token_id_offset must be nonnegative")
    numpy_generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        weight_decay=settings.weight_decay,
    )
    losses: list[float] = []
    for update in range(settings.updates):
        inputs, targets = sample_token_batch(
            token_ids,
            batch_size=settings.batch_size,
            context_length=model.context_length,
            generator=numpy_generator,
        )
        input_tensor = torch.as_tensor(inputs + token_id_offset, device=device)
        target_tensor = torch.as_tensor(targets + token_id_offset, device=device)
        learning_rate = cosine_learning_rate(
            update,
            settings.updates,
            maximum=settings.learning_rate,
            minimum=settings.minimum_learning_rate,
            warmup_steps=settings.warmup_updates,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        loss = torch_functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_tensor.reshape(-1),
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update}")
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if on_update is not None:
            on_update(update + 1, loss_value)
    return tuple(losses)
