"""Render the analysis figures used by the results notebook."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "token-row-transplants-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_row_transplants.plots import (  # noqa: E402
    cross_run_checkpoint_profiles,
    cross_run_source_loss,
    cross_run_transfer_spelling_effects,
    independent_initialization_spelling_effects,
    independent_initialization_word_surprisal,
    larger_decoder_assignment_effects,
    row_to_token_assignment_effects,
    row_to_token_assignment_persistence,
    save_figure,
    training_language_match_checkpoint_trajectories,
    training_language_match_final_perplexity,
    training_language_match_source_loss,
    training_language_match_spelling_effects,
)

FIGURE_NAMES = (
    "training_language_match_spelling_effects",
    "training_language_match_final_perplexity",
    "training_language_match_checkpoint_trajectories",
    "training_language_match_source_loss",
    "cross_run_transfer_spelling_effects",
    "cross_run_transfer_checkpoint_profiles",
    "cross_run_transfer_source_loss",
    "row_to_token_assignment_effects",
    "row_to_token_assignment_persistence",
    "larger_decoder_assignment_effects",
    "independent_initialization_spelling_effects",
    "independent_initialization_word_surprisal",
)

SOURCE_PATHS = (
    Path("results/training_language_match/seed_estimates.csv"),
    Path("results/training_language_match/performance.csv"),
    Path("results/training_language_match/performance_seed_values.csv"),
    Path("results/training_language_match/english_nll_checkpoints.csv"),
    Path("results/training_language_match/spelling_checkpoints.csv"),
    Path("results/cross_run_transfer/components.csv"),
    Path("results/cross_run_transfer/seed_estimates.csv"),
    Path("results/cross_run_transfer/checkpoint_profiles.csv"),
    Path("results/cross_run_transfer/source_language_costs.csv"),
    Path("results/row_to_token_assignment/components.csv"),
    Path("results/row_to_token_assignment/seed_estimates.csv"),
    Path("results/row_to_token_assignment/assignment_checkpoints.csv"),
    Path("results/larger_decoder_assignment/components.csv"),
    Path("results/larger_decoder_assignment/seed_estimates.csv"),
    Path("results/independent_initialization/condition_effects.csv"),
    Path("results/independent_initialization/condition_pair_estimates.csv"),
    Path("results/independent_initialization/word_surprisal.csv"),
)


def read_figure_tables(repository_root: Path) -> dict[str, pd.DataFrame]:
    """Read the compact tables shared by the notebook and figure command."""

    paths = {
        "training_seed_estimates": SOURCE_PATHS[0],
        "training_performance": SOURCE_PATHS[1],
        "training_performance_seeds": SOURCE_PATHS[2],
        "training_english_nll_checkpoints": SOURCE_PATHS[3],
        "training_spelling_checkpoints": SOURCE_PATHS[4],
        "cross_run_components": SOURCE_PATHS[5],
        "cross_run_seed_estimates": SOURCE_PATHS[6],
        "cross_run_checkpoint_profiles": SOURCE_PATHS[7],
        "cross_run_source_costs": SOURCE_PATHS[8],
        "assignment_components": SOURCE_PATHS[9],
        "assignment_seed_estimates": SOURCE_PATHS[10],
        "assignment_checkpoints": SOURCE_PATHS[11],
        "larger_decoder_components": SOURCE_PATHS[12],
        "larger_decoder_seed_estimates": SOURCE_PATHS[13],
        "initialization_condition_effects": SOURCE_PATHS[14],
        "initialization_condition_pairs": SOURCE_PATHS[15],
        "initialization_word_surprisal": SOURCE_PATHS[16],
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, relative_path in paths.items():
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing figure table: {relative_path}")
        table = pd.read_csv(path, keep_default_na=False)
        if table.empty:
            raise ValueError(f"empty figure table: {relative_path}")
        tables[name] = table
    return tables


def build_figures(tables: Mapping[str, pd.DataFrame]) -> dict[str, Figure]:
    """Build every analytical figure from the supplied result tables."""

    return {
        "training_language_match_spelling_effects": (
            training_language_match_spelling_effects(tables["training_seed_estimates"])
        ),
        "training_language_match_final_perplexity": (
            training_language_match_final_perplexity(
                tables["training_performance"],
                tables["training_performance_seeds"],
            )
        ),
        "training_language_match_checkpoint_trajectories": (
            training_language_match_checkpoint_trajectories(
                tables["training_english_nll_checkpoints"],
                tables["training_spelling_checkpoints"],
            )
        ),
        "training_language_match_source_loss": training_language_match_source_loss(
            tables["training_performance"],
            tables["training_performance_seeds"],
        ),
        "cross_run_transfer_spelling_effects": cross_run_transfer_spelling_effects(
            tables["cross_run_components"], tables["cross_run_seed_estimates"]
        ),
        "cross_run_transfer_checkpoint_profiles": cross_run_checkpoint_profiles(
            tables["cross_run_checkpoint_profiles"]
        ),
        "cross_run_transfer_source_loss": cross_run_source_loss(
            tables["cross_run_source_costs"]
        ),
        "row_to_token_assignment_effects": row_to_token_assignment_effects(
            tables["assignment_components"],
            tables["assignment_seed_estimates"],
        ),
        "row_to_token_assignment_persistence": row_to_token_assignment_persistence(
            tables["assignment_checkpoints"]
        ),
        "larger_decoder_assignment_effects": larger_decoder_assignment_effects(
            tables["larger_decoder_components"],
            tables["larger_decoder_seed_estimates"],
        ),
        "independent_initialization_spelling_effects": (
            independent_initialization_spelling_effects(
                tables["initialization_condition_effects"],
                tables["initialization_condition_pairs"],
            )
        ),
        "independent_initialization_word_surprisal": (
            independent_initialization_word_surprisal(
                tables["initialization_word_surprisal"]
            )
        ),
    }


def render(repository_root: Path, output_directory: Path) -> tuple[Path, ...]:
    """Render and save all analytical figures."""

    figures = build_figures(read_figure_tables(repository_root))
    if tuple(figures) != FIGURE_NAMES:
        raise ValueError("figure order does not match FIGURE_NAMES")
    output_paths: list[Path] = []
    for name, figure in figures.items():
        output_paths.extend(save_figure(figure, output_directory, name))
        plt.close(figure)
    return tuple(output_paths)


def verify_rendered(repository_root: Path, output_directory: Path) -> None:
    """Render a temporary copy and compare it with the committed figures."""

    with tempfile.TemporaryDirectory(prefix="token-row-transplants-figures-") as raw:
        temporary_directory = Path(raw)
        render(repository_root, temporary_directory)
        for name in FIGURE_NAMES:
            committed_png = output_directory / f"{name}.png"
            rendered_png = temporary_directory / f"{name}.png"
            committed_pdf = output_directory / f"{name}.pdf"
            if not committed_png.is_file() or not committed_pdf.is_file():
                raise FileNotFoundError(f"missing committed figure pair: {name}")
            committed = plt.imread(committed_png).astype(float)
            rendered = plt.imread(rendered_png).astype(float)
            height_difference = committed.shape[0] - rendered.shape[0]
            width_difference = committed.shape[1] - rendered.shape[1]
            if abs(height_difference) > 2 or abs(width_difference) > 2:
                raise ValueError(f"stale committed figure: {committed_png}")
            common_height = min(committed.shape[0], rendered.shape[0])
            common_width = min(committed.shape[1], rendered.shape[1])
            committed_top = max(height_difference // 2, 0)
            rendered_top = max(-height_difference // 2, 0)
            committed_left = max(width_difference // 2, 0)
            rendered_left = max(-width_difference // 2, 0)
            committed_crop = committed[
                committed_top : committed_top + common_height,
                committed_left : committed_left + common_width,
            ]
            rendered_crop = rendered[
                rendered_top : rendered_top + common_height,
                rendered_left : rendered_left + common_width,
            ]
            mean_difference = float(np.abs(committed_crop - rendered_crop).mean())
            if mean_difference > 0.01:
                raise ValueError(f"stale committed figure: {committed_png}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures",
        help="figure directory (default: REPOSITORY_ROOT/figures)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rerender figures and compare them with the committed PNGs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_directory = args.output_dir.resolve()
    if args.check:
        verify_rendered(ROOT, output_directory)
        print("Analysis figures are current")
        return 0
    for path in render(ROOT, output_directory):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
