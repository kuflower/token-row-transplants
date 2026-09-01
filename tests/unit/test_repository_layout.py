from __future__ import annotations

import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_NAMES = (
    "training_language_match",
    "cross_run_transfer",
    "row_to_token_assignment",
    "independent_initialization",
    "larger_decoder_assignment",
)


def test_project_name_and_layout() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["name"] == "token-row-transplants"
    assert "scripts" not in project
    assert project["urls"]["Repository"] == (
        "https://github.com/kuflower/token-row-transplants.git"
    )
    assert (REPOSITORY_ROOT / "src/token_row_transplants").is_dir()


def test_each_study_has_one_readable_configuration() -> None:
    for study_name in STUDY_NAMES:
        study = REPOSITORY_ROOT / "experiments" / study_name
        assert (study / "README.md").is_file()
        configurations = list(study.glob("*.toml"))
        assert configurations == [study / "experiment.toml"]
        with configurations[0].open("rb") as stream:
            payload = tomllib.load(stream)
        assert payload["study"]["name"] == study_name
