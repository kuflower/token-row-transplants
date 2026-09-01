"""Tokenized held-out contexts and active-block scoring."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tokenizers import Tokenizer

from .word_identity import FeatureContractError, normalize_word


class ProbeModel(Protocol):
    """Minimal runtime-visible interface needed for probe scoring."""

    training: bool

    def eval(self) -> Any:
        """Switch to evaluation mode."""

    def train(self, mode: bool = True) -> Any:
        """Restore the requested training mode."""

    def __call__(self, token_ids: Any) -> Any:
        """Return next-token logits for one token-ID batch."""


class ProbeError(ValueError):
    """Raised when probe contexts cannot be encoded safely."""


@dataclass(frozen=True, slots=True)
class ProbeScores:
    """Mean summed target log probability and context count by word."""

    values: tuple[tuple[str, float, int], ...]


def _target_token_ids(
    tokenizer: Tokenizer,
    word: str,
    *,
    vocabulary_size: int,
) -> tuple[int, ...]:
    """Encode one probe target exactly as whole-word scoring does."""
    target_ids = tuple(tokenizer.encode(" " + word).ids)
    if not target_ids:
        raise ProbeError(f"target word {word!r} has no tokenizer pieces")
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in target_ids):
        raise ProbeError(f"target word {word!r} encodes outside the vocabulary")
    return target_ids


def _target_piece_counts(
    words: Iterable[str],
    *,
    tokenizer: Tokenizer,
    vocabulary_size: int,
) -> dict[str, int]:
    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size < 1
    ):
        raise ProbeError("vocabulary size must be positive")
    counts: dict[str, int] = {}
    for word in words:
        if not isinstance(word, str):
            raise ProbeError("target piece-count words must be strings")
        try:
            normalized = normalize_word(word)
        except FeatureContractError as error:
            raise ProbeError(f"invalid target piece-count word: {error}") from error
        if word != normalized or any(character.isspace() for character in word):
            raise ProbeError(
                "target piece-count words must be normalized, whitespace-free "
                "identities"
            )
        if word in counts:
            raise ProbeError(f"duplicate target piece-count word: {word!r}")
        counts[word] = len(
            _target_token_ids(
                tokenizer,
                word,
                vocabulary_size=vocabulary_size,
            )
        )
    if not counts:
        raise ProbeError("target piece-count word set must not be empty")
    return {word: counts[word] for word in sorted(counts)}


def target_piece_counts(
    words: Iterable[str],
    *,
    tokenizer_path: Path,
    vocabulary_size: int,
) -> dict[str, int]:
    """Count pieces with the tokenizer used by the model.

    A target is encoded with the same leading ASCII space used by
    :meth:`EncodedProbe.from_jsonl`. Words must already be normalized and cannot
    contain whitespace.
    """
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return _target_piece_counts(
        words,
        tokenizer=tokenizer,
        vocabulary_size=vocabulary_size,
    )


class EncodedProbe:
    """Padded base-vocabulary probe contexts."""

    def __init__(
        self,
        *,
        words: tuple[str, ...],
        token_ids: np.ndarray,
        target_mask: np.ndarray,
        vocabulary_size: int,
    ) -> None:
        ids = np.asarray(token_ids, dtype=np.int64)
        mask = np.asarray(target_mask, dtype=bool)
        if ids.ndim != 2 or ids.shape != mask.shape or len(words) != len(ids):
            raise ProbeError(
                "probe words, token IDs, and masks have incompatible shapes"
            )
        if not len(ids) or ids.shape[1] < 2:
            raise ProbeError("probe must contain a sequence of at least two tokens")
        if np.any(ids < 0) or np.any(ids >= vocabulary_size):
            raise ProbeError("probe token ID lies outside the vocabulary")
        if np.any(mask[:, 0]) or np.any(mask.sum(axis=1) < 1):
            raise ProbeError("each probe row needs a predicted target token")
        if any(not word or word != word.strip() for word in words):
            raise ProbeError("probe words must be nonempty and trimmed")
        self.words = words
        self.token_ids = ids
        self.target_mask = mask
        self.vocabulary_size = vocabulary_size

    def context_counts(self) -> dict[str, int]:
        """Return the encoded context count for each probe word."""
        counts: dict[str, int] = {}
        for word in self.words:
            counts[word] = counts.get(word, 0) + 1
        return counts

    @classmethod
    def _from_json_lines(
        cls,
        *,
        lines: Iterable[str],
        tokenizer: Tokenizer,
        vocabulary_size: int,
        context_length: int,
        maximum_prefix_tokens: int,
        limit: int | None,
    ) -> EncodedProbe:
        words: list[str] = []
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for line_number, line in enumerate(lines, start=1):
            if limit is not None and len(rows) >= limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProbeError(f"invalid JSON on probe line {line_number}") from error
            if not isinstance(item, dict) or set(item) != {"prefix", "word"}:
                raise ProbeError(
                    f"probe line {line_number} must contain prefix and word"
                )
            prefix = item["prefix"]
            word = item["word"]
            if not isinstance(prefix, str) or not isinstance(word, str):
                raise ProbeError(f"probe line {line_number} fields must be strings")
            try:
                normalized_word = normalize_word(word)
            except FeatureContractError as error:
                raise ProbeError(f"invalid word on probe line {line_number}") from error
            if word != normalized_word or any(
                character.isspace() for character in word
            ):
                raise ProbeError(
                    f"probe line {line_number} word must be normalized and "
                    "whitespace-free"
                )
            if prefix != prefix.rstrip():
                raise ProbeError(
                    f"probe line {line_number} prefix must not end with whitespace"
                )
            prefix_ids = tokenizer.encode(prefix).ids[-maximum_prefix_tokens:]
            target_ids = tokenizer.encode(" " + word).ids
            if (
                not prefix_ids
                or not target_ids
                or len(prefix_ids) + len(target_ids) > context_length
            ):
                continue
            prefix_array = np.asarray(prefix_ids, dtype=np.int64)
            target_array = np.asarray(target_ids, dtype=np.int64)
            rows.append((prefix_array, target_array))
            words.append(word)
        if not rows:
            raise ProbeError("probe contains no encodable contexts")
        length = max(len(prefix) + len(target) for prefix, target in rows)
        ids = np.zeros((len(rows), length), dtype=np.int64)
        mask = np.zeros((len(rows), length), dtype=bool)
        for row, (prefix, target) in enumerate(rows):
            end = len(prefix) + len(target)
            ids[row, :end] = np.concatenate((prefix, target))
            mask[row, len(prefix) : end] = True
        return cls(
            words=tuple(words),
            token_ids=ids,
            target_mask=mask,
            vocabulary_size=vocabulary_size,
        )

    @classmethod
    def from_jsonl(
        cls,
        *,
        probe_path: Path,
        tokenizer_path: Path,
        vocabulary_size: int,
        context_length: int,
        maximum_prefix_tokens: int = 96,
        limit: int | None = None,
    ) -> EncodedProbe:
        """Encode a JSONL probe with one prefix and target word per row."""
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        with probe_path.open(encoding="utf-8") as handle:
            return cls._from_json_lines(
                lines=handle,
                tokenizer=tokenizer,
                vocabulary_size=vocabulary_size,
                context_length=context_length,
                maximum_prefix_tokens=maximum_prefix_tokens,
                limit=limit,
            )

    def score(
        self,
        model: ProbeModel,
        *,
        device: str,
        batch_size: int,
    ) -> ProbeScores:
        """Return the mean whole-word log probability for each probe word."""
        import torch
        import torch.nn.functional as functional

        if batch_size < 1:
            raise ProbeError("probe batch size must be positive")
        with torch.no_grad():
            was_training = model.training
            model.eval()
            try:
                scores = np.zeros(len(self.words), dtype=np.float64)
                for start in range(0, len(self.words), batch_size):
                    stop = start + batch_size
                    base_ids = torch.from_numpy(self.token_ids[start:stop]).to(device)
                    logits = model(base_ids)
                    if logits.shape[-1] != self.vocabulary_size:
                        raise ProbeError("model vocabulary does not match the probe")
                    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
                    gathered = log_probabilities[:, :-1].gather(
                        2, base_ids[:, 1:, None]
                    )[..., 0]
                    mask = torch.from_numpy(self.target_mask[start:stop, 1:]).to(device)
                    scores[start:stop] = (gathered * mask).sum(1).cpu().numpy()
            finally:
                model.train(was_training)

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for word, score in zip(self.words, scores, strict=True):
            sums[word] = sums.get(word, 0.0) + float(score)
            counts[word] = counts.get(word, 0) + 1
        return ProbeScores(
            tuple(
                (word, sums[word] / counts[word], counts[word]) for word in sorted(sums)
            )
        )


def probe_rows(
    english_updates: int,
    scores: ProbeScores,
) -> Iterable[tuple[object, ...]]:
    """Convert probe scores to rows for a checkpoint table."""
    for word, mean_log_probability, contexts in scores.values:
        yield (english_updates, word, mean_log_probability, contexts)
