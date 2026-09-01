from __future__ import annotations

import torch

from token_row_transplants.model import TiedDecoder
from token_row_transplants.transplant import transplant_token_rows


def _model(seed: int) -> TiedDecoder:
    torch.manual_seed(seed)
    return TiedDecoder(
        vocabulary_size=8,
        context_length=4,
        width=8,
        layers=1,
        heads=2,
    )


def test_transplant_copies_rows_and_keeps_the_receiving_body() -> None:
    receiving = _model(1).state_dict()
    donor = _model(2).state_dict()

    result = transplant_token_rows(receiving, donor)

    assert torch.equal(result["tok.weight"], donor["tok.weight"])
    assert torch.equal(result["head.weight"], donor["tok.weight"])
    for name in receiving:
        if name not in {"tok.weight", "head.weight"}:
            assert torch.equal(result[name], receiving[name])


def test_transplant_can_reassign_the_donor_rows() -> None:
    receiving = _model(3).state_dict()
    donor = _model(4).state_dict()
    assignment = (*range(1, 8), 0)

    result = transplant_token_rows(
        receiving,
        donor,
        source_row_for_token=assignment,
    )

    assert torch.equal(result["tok.weight"], donor["tok.weight"][list(assignment)])
