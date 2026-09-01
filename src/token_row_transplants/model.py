"""Small tied decoder used in the experiments."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        values = values + attended
        return values + cast(torch.Tensor, self.mlp(self.ln2(values)))


class TiedDecoder(nn.Module):
    """Decoder-only transformer with tied input and output token rows."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least two")
        if context_length < 2:
            raise ValueError("context_length must be at least two")
        if width < 1 or layers < 1 or heads < 1 or width % heads:
            raise ValueError("model dimensions are invalid")
        self.context_length = context_length
        self.tok = nn.Embedding(vocabulary_size, width)
        self.pos = nn.Embedding(context_length, width)
        self.blocks = nn.ModuleList(
            TransformerBlock(width, heads) for _ in range(layers)
        )
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocabulary_size, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._initialize)
        mask = torch.full((context_length, context_length), float("-inf")).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.context_length:
            raise ValueError(f"sequence length {length} exceeds {self.context_length}")
        values = self.tok(token_ids) + self.pos.weight[:length]
        mask = self.get_buffer("causal_mask")[:length, :length]
        for raw_block in self.blocks:
            block = cast(TransformerBlock, raw_block)
            values = block(values, mask)
        return cast(torch.Tensor, self.head(self.ln_f(values)))

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters())


def cosine_learning_rate(
    step: int,
    total_steps: int,
    *,
    maximum: float,
    minimum: float,
    warmup_steps: int,
) -> float:
    """Warm up linearly, then use cosine decay."""
    if step < 0 or total_steps < 1:
        raise ValueError("invalid learning-rate step")
    if not 0 <= minimum <= maximum:
        raise ValueError("learning-rate bounds are invalid")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if warmup_steps and step < warmup_steps:
        return maximum * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * progress))


class UntiedDecoder(TiedDecoder):
    """Decoder with separate input-embedding and output-projection rows."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__(
            vocabulary_size=vocabulary_size,
            context_length=context_length,
            width=width,
            layers=layers,
            heads=heads,
        )
        # Break the alias the parent installed, keeping the values, so an
        # untied model created from a tied parent starts numerically where
        # the tied one left off and differs only in what can move apart.
        self.head.weight = nn.Parameter(self.tok.weight.detach().clone())

    def roles_are_tied(self) -> bool:
        """Whether the two role matrices are still the same parameter."""
        return self.head.weight is self.tok.weight
