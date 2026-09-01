"""Copy tied token rows while preserving the receiving transformer body."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

TOKEN_MATRIX_KEY = "tok.weight"
OUTPUT_MATRIX_KEY = "head.weight"


class TransplantError(ValueError):
    """Raised when two model states cannot form a valid transplant."""


def _token_matrix(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> torch.Tensor:
    try:
        token_rows = state[TOKEN_MATRIX_KEY]
        output_rows = state[OUTPUT_MATRIX_KEY]
    except KeyError as error:
        raise TransplantError(f"{label} state has no tied token matrix") from error
    if token_rows.ndim != 2 or token_rows.shape != output_rows.shape:
        raise TransplantError(f"{label} token matrices have incompatible shapes")
    if not torch.equal(token_rows, output_rows):
        raise TransplantError(f"{label} input and output token rows are not tied")
    if not bool(torch.isfinite(token_rows).all()):
        raise TransplantError(f"{label} token matrix contains a non-finite value")
    return token_rows


def transplant_token_rows(
    receiving_state: Mapping[str, torch.Tensor],
    donor_state: Mapping[str, torch.Tensor],
    *,
    source_row_for_token: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Return the receiving state with the donor token matrix installed.

    By default the complete donor matrix is copied without changing token IDs.
    ``source_row_for_token`` can instead supply a permutation for the
    row-to-token assignment experiments.
    """

    receiving_rows = _token_matrix(receiving_state, label="receiving")
    donor_rows = _token_matrix(donor_state, label="donor")
    if receiving_rows.shape != donor_rows.shape:
        raise TransplantError("receiving and donor token matrices differ in shape")
    if set(receiving_state) != set(donor_state):
        raise TransplantError("receiving and donor states contain different parameters")

    installed_rows = donor_rows
    if source_row_for_token is not None:
        from .assignment import validate_reassignment

        assignment = validate_reassignment(source_row_for_token)
        if len(assignment) != donor_rows.shape[0]:
            raise TransplantError("row reassignment length differs from the vocabulary")
        row_ids = torch.tensor(assignment, device=donor_rows.device)
        installed_rows = donor_rows.index_select(0, row_ids)

    result = {name: value.detach().clone() for name, value in receiving_state.items()}
    copied_rows = installed_rows.to(
        device=receiving_rows.device,
        dtype=receiving_rows.dtype,
    )
    result[TOKEN_MATRIX_KEY].copy_(copied_rows)
    result[OUTPUT_MATRIX_KEY].copy_(copied_rows)
    return result
