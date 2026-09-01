"""Normalize committed notebook metadata and rich-display fallbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANONICAL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
RICH_MIME_TYPES = {"image/png", "text/html", "text/markdown"}


class NotebookNormalizationError(ValueError):
    """Raised when a file is not a supported notebook object."""


def normalize_notebook(path: Path) -> None:
    """Remove runtime-only metadata while preserving code and rich outputs."""

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        raise NotebookNormalizationError(f"invalid notebook: {path}")

    value["metadata"] = CANONICAL_METADATA
    for cell in value["cells"]:
        if not isinstance(cell, dict):
            raise NotebookNormalizationError(f"invalid cell in notebook: {path}")
        metadata = cell.get("metadata")
        if not isinstance(metadata, dict):
            raise NotebookNormalizationError(f"invalid cell metadata: {path}")
        metadata.pop("execution", None)
        metadata.pop("trusted", None)

        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise NotebookNormalizationError(f"invalid cell outputs: {path}")
        for output in outputs:
            if not isinstance(output, dict):
                raise NotebookNormalizationError(f"invalid output in notebook: {path}")
            data = output.get("data")
            if isinstance(data, dict) and RICH_MIME_TYPES.intersection(data):
                data.pop("text/plain", None)

    payload = json.dumps(value, ensure_ascii=False, indent=1) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    return parser


def main() -> int:
    """Normalize notebooks named on the command line."""

    for path in _parser().parse_args().notebooks:
        normalize_notebook(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
