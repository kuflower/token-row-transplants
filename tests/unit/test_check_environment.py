from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from scripts.check_environment import (
    EnvironmentCheckError,
    environment_problems,
    expected_project_distribution,
    expected_python_version,
    pinned_requirements,
)


def _write_environment_files(root: Path) -> None:
    (root / ".python-version").write_text("3.12.14\n", encoding="utf-8")
    (root / "requirements-lock.txt").write_text(
        "numpy==2.5.1\nscipy==1.18.0\n", encoding="utf-8"
    )
    (root / "requirements-ci.txt").write_text(
        "-r requirements-lock.txt\npytest==9.0.3\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "token-row-transplants"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )


def test_environment_files_define_exact_versions(tmp_path: Path) -> None:
    _write_environment_files(tmp_path)

    assert expected_python_version(tmp_path) == "3.12.14"
    assert expected_project_distribution(tmp_path) == (
        "token-row-transplants",
        "1.0.0",
    )
    assert pinned_requirements(tmp_path / "requirements-ci.txt") == {
        "numpy": "2.5.1",
        "pytest": "9.0.3",
        "scipy": "1.18.0",
    }


def test_environment_problems_report_interpreter_and_package_drift(
    tmp_path: Path,
) -> None:
    _write_environment_files(tmp_path)
    installed = {
        "numpy": "2.3.5",
        "pytest": "9.0.3",
        "token-row-transplants": "0.0.9",
    }

    def installed_version(package_name: str) -> str:
        try:
            return installed[package_name]
        except KeyError as error:
            raise PackageNotFoundError(package_name) from error

    assert environment_problems(
        tmp_path,
        python_version="3.12.11",
        installed_version=installed_version,
    ) == [
        "Python: expected 3.12.14, found 3.12.11",
        "token-row-transplants: expected 1.0.0, found 0.0.9",
        "numpy: expected 2.5.1, found 2.3.5",
        "scipy: expected 1.18.0, not installed",
    ]


def test_unpinned_requirement_is_rejected(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("scipy>=1.14\n", encoding="utf-8")

    with pytest.raises(EnvironmentCheckError, match="not an exact package pin"):
        pinned_requirements(requirements)
