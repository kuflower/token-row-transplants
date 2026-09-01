"""Check that the active Python environment matches the repository pins."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EXACT_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^]]+\])?==(?P<version>[^;\s]+)$"
)


class EnvironmentCheckError(RuntimeError):
    """Raised when a checked-in environment file is invalid."""


def expected_python_version(repository_root: Path) -> str:
    """Read the exact interpreter version used for development and CI."""

    version_file = repository_root / ".python-version"
    lines = [
        line.strip()
        for line in version_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1 or not re.fullmatch(r"\d+\.\d+\.\d+", lines[0]):
        raise EnvironmentCheckError(
            f"{version_file} must contain one exact X.Y.Z Python version"
        )
    return lines[0]


def pinned_requirements(requirements_file: Path) -> dict[str, str]:
    """Read exact pins, following local ``-r`` includes."""

    pins: dict[str, str] = {}
    visited: set[Path] = set()

    def read_file(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        for line_number, raw_line in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r ") or line.startswith("--requirement "):
                included_name = line.split(maxsplit=1)[1]
                read_file(resolved.parent / included_name)
                continue
            match = EXACT_REQUIREMENT.fullmatch(line)
            if match is None:
                raise EnvironmentCheckError(
                    f"{resolved}:{line_number} is not an exact package pin: {line}"
                )
            package_name = match.group("name").lower().replace("_", "-")
            package_version = match.group("version")
            previous = pins.get(package_name)
            if previous is not None and previous != package_version:
                raise EnvironmentCheckError(
                    f"conflicting pins for {package_name}: {previous} and "
                    f"{package_version}"
                )
            pins[package_name] = package_version

    read_file(requirements_file)
    return pins


def expected_project_distribution(repository_root: Path) -> tuple[str, str]:
    """Read the distribution name and version from ``pyproject.toml``."""

    pyproject_path = repository_root / "pyproject.toml"
    with pyproject_path.open("rb") as stream:
        payload = tomllib.load(stream)
    project = payload.get("project")
    if not isinstance(project, dict):
        raise EnvironmentCheckError(f"{pyproject_path} has no [project] table")
    name = project.get("name")
    project_version = project.get("version")
    if not isinstance(name, str) or not name.strip():
        raise EnvironmentCheckError(f"{pyproject_path} has no project name")
    if not isinstance(project_version, str) or not project_version.strip():
        raise EnvironmentCheckError(f"{pyproject_path} has no project version")
    return name, project_version


def environment_problems(
    repository_root: Path,
    *,
    python_version: str,
    installed_version: Callable[[str], str] = version,
    python_only: bool = False,
) -> list[str]:
    """Return readable differences between the active and checked-in environment."""

    required_python = expected_python_version(repository_root)
    problems = []
    if python_version != required_python:
        problems.append(f"Python: expected {required_python}, found {python_version}")
    if python_only:
        return problems

    project_name, required_project_version = expected_project_distribution(
        repository_root
    )
    try:
        current_project_version = installed_version(project_name)
    except PackageNotFoundError:
        problems.append(
            f"{project_name}: expected {required_project_version}, not installed"
        )
    else:
        if current_project_version != required_project_version:
            problems.append(
                f"{project_name}: expected {required_project_version}, "
                f"found {current_project_version}"
            )

    requirements_file = repository_root / "requirements-ci.txt"
    for package_name, required_version in sorted(
        pinned_requirements(requirements_file).items()
    ):
        try:
            current_version = installed_version(package_name)
        except PackageNotFoundError:
            problems.append(
                f"{package_name}: expected {required_version}, not installed"
            )
            continue
        if current_version != required_version:
            problems.append(
                f"{package_name}: expected {required_version}, found {current_version}"
            )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="token-row-transplants checkout (default: parent of this script)",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="check only the interpreter before creating a virtual environment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    try:
        problems = environment_problems(
            repository_root,
            python_version=".".join(str(part) for part in sys.version_info[:3]),
            python_only=args.python_only,
        )
    except (EnvironmentCheckError, OSError) as error:
        print(f"environment check failed: {error}", file=sys.stderr)
        return 2
    if problems:
        print("environment check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "Run `make environment` with the Python version in .python-version, "
            "activate .venv, and retry.",
            file=sys.stderr,
        )
        return 1
    print("environment matches .python-version and requirements-ci.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
