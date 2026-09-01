from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import token_row_transplants


def test_package_metadata() -> None:
    package_root = Path(token_row_transplants.__file__).resolve().parent

    assert package_root.name == "token_row_transplants"
    assert token_row_transplants.__all__ == ["__version__"]
    assert token_row_transplants.__version__ == "1.0.0"


def test_package_import_has_no_optional_dependencies() -> None:
    script = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    optional = ("matplotlib", "pandas", "scipy", "tokenizers", "torch")
    if name == optional or name.startswith(tuple(item + "." for item in optional)):
        raise AssertionError(f"package imported optional dependency: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import token_row_transplants
assert not {"matplotlib", "pandas", "scipy", "tokenizers", "torch"} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
