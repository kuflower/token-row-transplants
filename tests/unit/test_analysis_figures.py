from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from scripts import render_analysis_figures

from token_row_transplants import plots

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FIGURES = {
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
}


def test_plot_palette_uses_color_for_language_and_shape_for_outcome() -> None:
    language_colors = {str(style["color"]) for style in plots.LANGUAGE_STYLE.values()}
    assert len(language_colors) == 3
    assert plots.SPANISH == "#9A4D00"
    assert plots.GERMAN == "#1F4E79"

    before = plots.OUTCOME_STYLE["step_zero"]
    across = plots.OUTCOME_STYLE["learning_curve_auc"]
    assert before["marker"] != across["marker"]
    assert before["filled"] is True
    assert across["filled"] is False
    assert "color" not in before
    assert "color" not in across

    assert plots.CONDITION_STYLE["english"]["color"] == plots.ENGLISH
    assert plots.CONDITION_STYLE["same_language"]["color"] == plots.MATCH_COLOR
    assert plots.CONDITION_STYLE["other_language"]["color"] == plots.MISMATCH_COLOR
    assert plots.MISMATCH_COLOR != plots.GERMAN
    assert len({style["marker"] for style in plots.CONDITION_STYLE.values()}) == len(
        plots.CONDITION_STYLE
    )


def test_figure_sources_exist_and_load() -> None:
    assert len(render_analysis_figures.SOURCE_PATHS) == 17
    assert len(set(render_analysis_figures.SOURCE_PATHS)) == 17
    assert all(
        (REPOSITORY_ROOT / relative_path).is_file()
        for relative_path in render_analysis_figures.SOURCE_PATHS
    )

    tables = render_analysis_figures.read_figure_tables(REPOSITORY_ROOT)

    assert len(tables) == len(render_analysis_figures.SOURCE_PATHS)
    assert all(not table.empty for table in tables.values())


def test_all_analysis_figures_build_from_result_tables() -> None:
    tables = render_analysis_figures.read_figure_tables(REPOSITORY_ROOT)
    figures = render_analysis_figures.build_figures(tables)
    try:
        assert tuple(figures) == render_analysis_figures.FIGURE_NAMES
        assert set(figures) == EXPECTED_FIGURES
        assert all(isinstance(figure, Figure) for figure in figures.values())
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_seed_tables_match_the_published_summaries() -> None:
    tables = render_analysis_figures.read_figure_tables(REPOSITORY_ROOT)
    performance = tables["training_performance"]
    performance_seeds = tables["training_performance_seeds"]
    for row in performance.itertuples(index=False):
        selected = performance_seeds.loc[
            performance_seeds["receiving_body_language"].eq(row.receiving_body_language)
            & performance_seeds["installed_rows"].eq(row.installed_rows)
        ]
        assert len(selected) == 12
        perplexity = selected["final_english_perplexity"].to_numpy(dtype=float)
        source_loss = selected["source_language_nll_increase"].to_numpy(dtype=float)
        assert float(np.exp(np.log(perplexity).mean())) == pytest.approx(
            row.final_english_perplexity
        )
        assert float(source_loss.mean()) == pytest.approx(
            row.source_language_nll_increase
        )

    condition_summaries = tables["initialization_condition_effects"]
    condition_pairs = tables["initialization_condition_pairs"]
    for row in condition_summaries.itertuples(index=False):
        selected = condition_pairs.loc[
            condition_pairs["condition"].eq(row.condition)
            & condition_pairs["language"].eq(row.language)
            & condition_pairs["outcome"].eq(row.outcome),
            "estimate",
        ].to_numpy(dtype=float)
        assert len(selected) == 12
        assert float(selected.mean()) == pytest.approx(row.estimate)
        assert float(selected.std(ddof=1)) == pytest.approx(row.sample_sd)

    larger_components = tables["larger_decoder_components"]
    larger_seeds = tables["larger_decoder_seed_estimates"]
    for row in larger_components.itertuples(index=False):
        selected = larger_seeds.loc[
            larger_seeds["contrast_id"].eq(row.contrast_id)
            & larger_seeds["language"].eq(row.language)
            & larger_seeds["outcome"].eq(row.outcome),
            "estimate",
        ].to_numpy(dtype=float)
        assert len(selected) == 14
        assert float(selected.mean()) == pytest.approx(row.estimate)


def test_english_reference_legend_includes_its_value() -> None:
    tables = render_analysis_figures.read_figure_tables(REPOSITORY_ROOT)
    performance = tables["training_performance"]
    reference = performance.loc[
        performance["receiving_body_language"].eq("english")
        & performance["installed_rows"].eq("english_trained"),
        "final_english_perplexity",
    ].item()
    figure = plots.training_language_match_final_perplexity(
        performance,
        tables["training_performance_seeds"],
    )
    try:
        legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert legend_labels[-1] == f"English-trained reference ({reference:.2f})"
    finally:
        plt.close(figure)


def test_assignment_legend_only_identifies_the_three_reassignments() -> None:
    tables = render_analysis_figures.read_figure_tables(REPOSITORY_ROOT)
    figure = plots.row_to_token_assignment_effects(
        tables["assignment_components"],
        tables["assignment_seed_estimates"],
    )
    try:
        legend = figure.legends[0]
        assert [text.get_text() for text in legend.get_texts()] == [
            "Reassignment 1",
            "Reassignment 2",
            "Reassignment 3",
        ]
        assert [handle.get_marker() for handle in legend.legend_handles] == [
            "o",
            "s",
            "^",
        ]
        row_labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        assert all("\n" not in label for label in row_labels)
    finally:
        plt.close(figure)


def test_render_writes_pdf_and_png_pairs(tmp_path: Path) -> None:
    paths = render_analysis_figures.render(REPOSITORY_ROOT, tmp_path)
    expected_paths = {
        tmp_path / f"{name}.{extension}"
        for name in EXPECTED_FIGURES
        for extension in ("pdf", "png")
    }

    assert set(paths) == expected_paths
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_committed_figures_match_the_renderer() -> None:
    render_analysis_figures.verify_rendered(
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "figures",
    )
