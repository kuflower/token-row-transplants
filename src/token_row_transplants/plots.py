"""Plots used in the results notebook."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "token-row-transplants-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SOURCE_LANGUAGES = ("spanish", "german")
LANGUAGE_LABEL = {
    "spanish": "Spanish",
    "german": "German",
    "english": "English",
}
BODY_PANEL_TITLE = {
    "spanish": "Spanish-trained body",
    "german": "German-trained body",
}

INK = "#2B2B2B"
MUTED = "#67615C"
GRID = "#DDDAD6"
LIGHT_GRAY = "#F1F0EE"
INTERVAL_BOUNDARY = "#918B86"

SPANISH = "#9A4D00"
GERMAN = "#1F4E79"
ENGLISH = "#7A3E65"

MATCH_COLOR = "#2F7D32"
MISMATCH_COLOR = "#5B4E9C"
INITIAL_ROWS = "#8A817A"

LANGUAGE_STYLE: Mapping[str, Mapping[str, str]] = {
    "spanish": {"color": SPANISH, "marker": "o", "linestyle": "-"},
    "german": {"color": GERMAN, "marker": "s", "linestyle": "--"},
    "english": {"color": ENGLISH, "marker": "^", "linestyle": "-."},
}

T_95_DF_11 = 2.200985160082949
T_SIMULTANEOUS_95_DF_11 = 3.4966141732536835

PLOT_STYLE: dict[Any, Any] = {
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.titleweight": "normal",
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.titlesize": 10.8,
    "font.family": "DejaVu Sans",
    "font.size": 9.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.fontsize": 8.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 180,
    "text.color": INK,
    "xtick.color": MUTED,
    "xtick.labelsize": 8.8,
    "ytick.color": MUTED,
    "ytick.labelsize": 8.8,
}

OUTCOME_STYLE: Mapping[str, Mapping[str, object]] = {
    "step_zero": {
        "filled": True,
        "label": "Before English continuation",
        "marker": "D",
        "offset": -0.09,
    },
    "learning_curve_auc": {
        "filled": False,
        "label": "Across English continuation",
        "marker": "o",
        "offset": 0.09,
    },
}

CONDITION_STYLE: Mapping[str, Mapping[str, str]] = {
    "reference": {"color": INK, "marker": "D"},
    "initial": {"color": INITIAL_ROWS, "marker": "^"},
    "english": {"color": ENGLISH, "marker": "o"},
    "same_language": {"color": MATCH_COLOR, "marker": "P"},
    "other_language": {"color": MISMATCH_COLOR, "marker": "s"},
}


@dataclass(frozen=True)
class SourceCostEstimate:
    condition: str
    label: str
    mean: float
    low: float
    high: float
    seed_values: tuple[float, ...]


def _student_interval(
    values: Sequence[float], *, critical_value: float = T_95_DF_11
) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 2 or not np.all(np.isfinite(sample)):
        raise ValueError("an interval needs at least two finite values")
    mean = float(sample.mean())
    half_width = critical_value * float(sample.std(ddof=1)) / math.sqrt(len(sample))
    return mean, mean - half_width, mean + half_width


def _one_row(
    frame: pd.DataFrame,
    **filters: object,
) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        description = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise ValueError(f"expected one result row for {description}")
    return selected.iloc[0]


def _seed_values(
    frame: pd.DataFrame,
    *,
    contrast_id: str,
    language: str,
    outcome: str,
    expected_count: int = 12,
) -> dict[int, float]:
    selected = frame.loc[
        frame["contrast_id"].eq(contrast_id)
        & frame["language"].eq(language)
        & frame["outcome"].eq(outcome)
    ]
    values = {
        int(row.seed): float(row.estimate)
        for row in selected[["seed", "estimate"]].itertuples(index=False)
    }
    if len(values) != len(selected) or len(values) != expected_count:
        raise ValueError(
            f"expected {expected_count} seed estimates for "
            f"{contrast_id}/{language}/{outcome}"
        )
    return values


def _pair_values(
    frame: pd.DataFrame,
    *,
    condition: str,
    language: str,
    outcome: str,
) -> tuple[float, ...]:
    selected = frame.loc[
        frame["condition"].eq(condition)
        & frame["language"].eq(language)
        & frame["outcome"].eq(outcome)
    ].sort_values("pair_id")
    if len(selected) != 12 or selected["pair_id"].nunique() != 12:
        raise ValueError(
            f"expected 12 initialization pairs for {condition}/{language}/{outcome}"
        )
    return tuple(float(value) for value in selected["estimate"])


def _interval_point(
    ax: Axes,
    *,
    estimate: float,
    low: float,
    high: float,
    position: float,
    color: str,
    marker: str,
    filled: bool = True,
    horizontal: bool = True,
    marker_size: float = 5.2,
) -> None:
    error = np.asarray([[estimate - low], [high - estimate]])
    if horizontal:
        ax.errorbar(
            estimate,
            position,
            xerr=error,
            color=color,
            marker=marker,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=marker_size,
            capsize=2.4,
            linewidth=1.25,
            zorder=3,
        )
    else:
        ax.errorbar(
            position,
            estimate,
            yerr=error,
            color=color,
            marker=marker,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            markersize=marker_size,
            capsize=2.4,
            linewidth=1.25,
            zorder=3,
        )


def _seed_points(
    ax: Axes,
    *,
    values: Sequence[float],
    position: float,
    color: str,
    horizontal: bool = True,
    half_span: float = 0.035,
    marker: str = "o",
    size: float = 11,
    alpha: float = 0.22,
) -> None:
    """Draw deterministic seed or pair estimates behind a summary interval."""

    estimates = np.asarray(values, dtype=float)
    if estimates.ndim != 1 or len(estimates) < 2 or not np.all(np.isfinite(estimates)):
        raise ValueError("seed points need at least two finite estimates")
    offsets = np.linspace(-half_span, half_span, len(estimates))
    if horizontal:
        x, y = estimates, position + offsets
    else:
        x, y = position + offsets, estimates
    ax.scatter(
        x,
        y,
        color=color,
        edgecolor="none",
        marker=marker,
        s=size,
        alpha=alpha,
        zorder=2,
    )


def _mean_label(
    ax: Axes,
    *,
    estimate: float,
    position: float,
    color: str = MUTED,
) -> None:
    """Place a compact two-decimal label above a horizontal point estimate."""

    ax.annotate(
        f"{estimate:.2f}",
        xy=(estimate, position),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=color,
        fontsize=7.3,
        zorder=4,
        clip_on=False,
        bbox={
            "facecolor": ax.get_facecolor(),
            "edgecolor": "none",
            "pad": 0.15,
            "alpha": 0.9,
        },
    )


def _outcome_legend(*, color: str = INK) -> tuple[Line2D, ...]:
    handles = []
    for outcome in ("step_zero", "learning_curve_auc"):
        style = OUTCOME_STYLE[outcome]
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle="none",
                marker=str(style["marker"]),
                markerfacecolor=color if bool(style["filled"]) else "white",
                markeredgecolor=color,
                label=str(style["label"]),
            )
        )
    return tuple(handles)


def save_figure(figure: Figure, output_directory: Path, name: str) -> tuple[Path, Path]:
    """Save one figure in the two formats used by the repository."""

    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_path = output_directory / f"{name}.pdf"
    png_path = output_directory / f"{name}.png"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(png_path, bbox_inches="tight", dpi=180)
    return pdf_path, png_path


def training_language_match_spelling_effects(
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Compare matched rows with a different row language or body language."""

    contrasts = {
        "spanish": {
            "matched": "spanish_matched_benefit",
            "selectivity": "spanish_over_german_rows_on_spanish_body",
            "other_body": "spanish_rows_on_german_body",
        },
        "german": {
            "matched": "german_matched_benefit",
            "selectivity": "german_over_spanish_rows_on_german_body",
            "other_body": "german_rows_on_spanish_body",
        },
    }
    condition_labels = (
        "Matched rows\nand body",
        "Other-language rows\nbody unchanged",
        "Other-language body\nrows unchanged",
    )
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.5, 3.2), sharey=True, constrained_layout=True
        )
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            positions = np.arange(3, dtype=float)
            for outcome in ("step_zero", "learning_curve_auc"):
                style = OUTCOME_STYLE[outcome]
                matched = _seed_values(
                    seed_estimates,
                    contrast_id=contrasts[language]["matched"],
                    language=language,
                    outcome=outcome,
                )
                selectivity = _seed_values(
                    seed_estimates,
                    contrast_id=contrasts[language]["selectivity"],
                    language=language,
                    outcome=outcome,
                )
                other_body = _seed_values(
                    seed_estimates,
                    contrast_id=contrasts[language]["other_body"],
                    language=language,
                    outcome=outcome,
                )
                values = np.asarray(
                    [
                        (
                            matched[seed],
                            matched[seed] - selectivity[seed],
                            other_body[seed],
                        )
                        for seed in sorted(matched)
                    ]
                )
                means = values.mean(axis=0)
                half_width = (
                    T_95_DF_11 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
                )
                x = positions + cast(float, style["offset"])
                color = str(LANGUAGE_STYLE[language]["color"])
                for point_position, seed_column in zip(x, values.T, strict=True):
                    _seed_points(
                        ax,
                        values=seed_column,
                        position=float(point_position),
                        color=color,
                        horizontal=False,
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=half_width,
                    color=color,
                    marker=str(style["marker"]),
                    markerfacecolor=color if bool(style["filled"]) else "white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    linestyle="none",
                    markersize=5.2,
                    capsize=2.4,
                    linewidth=1.25,
                    zorder=3,
                )
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xticks(positions, condition_labels)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_title(
                f"({chr(97 + panel_index)}) {LANGUAGE_LABEL[language]} spelling effect"
            )
            ax.set_ylim(-0.075, 0.36)
            ax.set_yticks((0, 0.1, 0.2, 0.3))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        axes[0].set_ylabel(
            "Change in spelling effect from initial rows\n(standardized units)"
        )
        figure.legend(
            handles=_outcome_legend(),
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        return figure


def training_language_match_final_perplexity(
    performance: pd.DataFrame,
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Plot final English perplexity for source- and English-trained models."""

    row_sources = {
        "source": {"marker": "o", "filled": True},
        "english": {"marker": "s", "filled": False},
    }
    reference = _one_row(
        performance,
        receiving_body_language="english",
        installed_rows="english_trained",
    )
    reference_perplexity = float(reference["final_english_perplexity"])
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.5, 3.15), sharex=True, constrained_layout=True
        )
        plotted_values: list[float] = []
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            source_rows = f"{language}_trained"
            for body_language, center in (
                (language, 1.0),
                ("english", 0.0),
            ):
                for rows_key, rows_name, offset in (
                    ("source", source_rows, 0.14),
                    ("english", "english_trained", -0.14),
                ):
                    row = _one_row(
                        performance,
                        receiving_body_language=body_language,
                        installed_rows=rows_name,
                    )
                    estimate = float(row["final_english_perplexity"])
                    low = float(row["final_english_perplexity_ci95_low"])
                    high = float(row["final_english_perplexity_ci95_high"])
                    style = row_sources[rows_key]
                    color = (
                        str(LANGUAGE_STYLE[language]["color"])
                        if rows_key == "source"
                        else ENGLISH
                    )
                    seed_values = seed_estimates.loc[
                        seed_estimates["receiving_body_language"].eq(body_language)
                        & seed_estimates["installed_rows"].eq(rows_name),
                        "final_english_perplexity",
                    ].to_numpy(dtype=float)
                    if len(seed_values) != 12:
                        raise ValueError(
                            "expected 12 final-perplexity seeds for "
                            f"{body_language}/{rows_name}"
                        )
                    _seed_points(
                        ax,
                        values=seed_values,
                        position=center + offset,
                        color=color,
                    )
                    _interval_point(
                        ax,
                        estimate=estimate,
                        low=low,
                        high=high,
                        position=center + offset,
                        color=color,
                        marker=str(style["marker"]),
                        filled=bool(style["filled"]),
                    )
                    if not (body_language == "english" and rows_key == "english"):
                        _mean_label(
                            ax,
                            estimate=estimate,
                            position=center + offset,
                            color=color,
                        )
                    plotted_values.extend((low, high))
            ax.axvline(
                reference_perplexity,
                color=ENGLISH,
                linestyle=(0, (1.5, 2.2)),
                linewidth=0.8,
                alpha=0.7,
            )
            source_name = LANGUAGE_LABEL[language]
            ax.set_yticks(
                (1.0, 0.0),
                (f"{source_name}-trained body", "English-trained body"),
            )
            ax.set_ylim(-0.45, 1.45)
            ax.set_title(f"({chr(97 + panel_index)}) {source_name}-source conditions")
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.set_xlabel("Final held-out English perplexity\n(lower is better)")
        low, high = min(plotted_values), max(plotted_values)
        padding = 0.08 * (high - low)
        for ax in axes:
            ax.set_xlim(low - padding, high + 1.5 * padding)
        figure.legend(
            handles=(
                Line2D(
                    [],
                    [],
                    color=SPANISH,
                    marker="o",
                    linestyle="none",
                    label="Spanish rows",
                ),
                Line2D(
                    [],
                    [],
                    color=GERMAN,
                    marker="o",
                    linestyle="none",
                    label="German rows",
                ),
                Line2D(
                    [],
                    [],
                    color=ENGLISH,
                    marker="s",
                    markerfacecolor="white",
                    linestyle="none",
                    label="English rows",
                ),
                Line2D(
                    [],
                    [],
                    color=ENGLISH,
                    linestyle=(0, (1.5, 2.2)),
                    label=f"English-trained reference ({reference_perplexity:.2f})",
                ),
            ),
            loc="outside lower center",
            ncol=4,
            frameon=False,
        )
        return figure


def training_language_match_checkpoint_trajectories(
    english_loss_trajectory: pd.DataFrame,
    spelling_trajectory: pd.DataFrame,
) -> Figure:
    """Plot the two Experiment 1 checkpoint summaries used in the notebook."""

    loss = english_loss_trajectory.copy()
    loss["estimate"] = -loss["english_rows_minus_source_rows_nll"]
    loss["low"] = -loss["simultaneous95_high"]
    loss["high"] = -loss["simultaneous95_low"]
    spelling = spelling_trajectory.copy()
    spelling["estimate"] = spelling["initial_minus_source_spelling_slope"]
    spelling = spelling.rename(
        columns={
            "simultaneous95_low": "low",
            "simultaneous95_high": "high",
        }
    )
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.25), constrained_layout=True)
        panels = (
            (
                axes[0],
                loss,
                "receiving_body_language",
                "(a) Source-row minus English-row NLL",
                "Source-row NLL minus English-row NLL\n(nats per token)",
            ),
            (
                axes[1],
                spelling,
                "language",
                "(b) Source-row effect on the English-trained body",
                "Change in spelling effect from initial rows\n(standardized units)",
            ),
        )
        for ax, table, language_column, title, ylabel in panels:
            checkpoints = sorted(
                int(value) for value in table["english_updates"].unique()
            )
            x = np.log1p(np.asarray(checkpoints, dtype=float))
            labeled_checkpoints = (0, 50, 200, 800, 6_100)
            marker_indices = [
                index
                for index, checkpoint in enumerate(checkpoints)
                if checkpoint in labeled_checkpoints
            ]
            for language in SOURCE_LANGUAGES:
                language_style = LANGUAGE_STYLE[language]
                selected = table.loc[table[language_column].eq(language)].sort_values(
                    "english_updates"
                )
                if list(selected["english_updates"].astype(int)) != checkpoints:
                    raise ValueError(f"incomplete checkpoint trajectory for {language}")
                estimate = selected["estimate"].to_numpy(dtype=float)
                low = selected["low"].to_numpy(dtype=float)
                high = selected["high"].to_numpy(dtype=float)
                color = str(language_style["color"])
                ax.fill_between(
                    x, low, high, color=color, alpha=0.10, linewidth=0, zorder=1
                )
                ax.plot(
                    x,
                    estimate,
                    color=color,
                    marker=str(language_style["marker"]),
                    markevery=marker_indices,
                    markersize=2.8,
                    markeredgewidth=0.6,
                    linestyle=str(language_style["linestyle"]),
                    linewidth=1.45,
                    label=LANGUAGE_LABEL[language],
                    zorder=2,
                )
            ax.axhline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
            ax.set_xticks(
                np.log1p(labeled_checkpoints),
                ("0", "50", "200", "800", "6,100"),
            )
            ax.set_xlabel("English updates (log(1 + updates) scale)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside lower center", ncol=2, frameon=False
        )
        return figure


def _source_cost_plot(
    estimates: Mapping[str, Sequence[SourceCostEstimate]],
) -> Figure:
    interval_limits = [
        bound
        for rows in estimates.values()
        for estimate in rows
        for bound in (estimate.low, estimate.high)
    ]
    left = min(0.0, min(interval_limits))
    right = max(interval_limits)
    padding = max(0.08 * (right - left), 0.05)
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.25), sharex=True, sharey=True)
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            rows = estimates[language]
            positions = np.arange(len(rows), dtype=float)
            ax.grid(axis="x")
            ax.axvline(0, color=MUTED, linewidth=0.8)
            reference = next(row for row in rows if row.condition == "reference")
            ax.axvline(
                reference.mean,
                color=INK,
                linestyle=(0, (1.5, 2.2)),
                linewidth=0.8,
                alpha=0.65,
            )
            for position, estimate in zip(positions, rows, strict=True):
                style = CONDITION_STYLE[estimate.condition]
                color = str(style["color"])
                _seed_points(
                    ax,
                    values=estimate.seed_values,
                    position=float(position),
                    color=color,
                )
                _interval_point(
                    ax,
                    estimate=estimate.mean,
                    low=estimate.low,
                    high=estimate.high,
                    position=float(position),
                    color=color,
                    marker=str(style["marker"]),
                )
                _mean_label(
                    ax,
                    estimate=estimate.mean,
                    position=float(position),
                    color=color,
                )
            ax.set_yticks(positions, [row.label for row in rows])
            ax.set_ylim(len(rows) - 0.4, -0.6)
            ax.set_xlim(left - padding, right + padding)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.tick_params(axis="y", labelsize=8.1, pad=4)
            ax.set_axisbelow(True)
        figure.supxlabel(
            "Increase in held-out source NLL after English continuation\n"
            "(nats per token, lower is better)",
            fontsize=9.5,
            y=0.015,
        )
        figure.subplots_adjust(
            left=0.24, right=0.985, bottom=0.25, top=0.88, wspace=0.12
        )
        return figure


def training_language_match_source_loss(
    performance: pd.DataFrame,
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Plot source-language loss after Experiment 1 English continuation."""

    estimates: dict[str, list[SourceCostEstimate]] = {}
    for language in SOURCE_LANGUAGES:
        other_language = next(item for item in SOURCE_LANGUAGES if item != language)
        installed_rows = (
            ("reference", "Source-language rows\n(reference)", f"{language}_trained"),
            ("initial", "Initial rows", "initial"),
            ("english", "English rows", "english_trained"),
            (
                "other_language",
                "Other-language rows",
                f"{other_language}_trained",
            ),
        )
        selected = performance.loc[performance["receiving_body_language"].eq(language)]
        rows = []
        for condition, label, row_language in installed_rows:
            row = _one_row(selected, installed_rows=row_language)
            seed_values = seed_estimates.loc[
                seed_estimates["receiving_body_language"].eq(language)
                & seed_estimates["installed_rows"].eq(row_language),
                "source_language_nll_increase",
            ].to_numpy(dtype=float)
            if len(seed_values) != 12:
                raise ValueError(
                    f"expected 12 source-loss seeds for {language}/{row_language}"
                )
            rows.append(
                SourceCostEstimate(
                    condition=condition,
                    label=label,
                    mean=float(row["source_language_nll_increase"]),
                    low=float(row["source_language_nll_increase_ci95_low"]),
                    high=float(row["source_language_nll_increase_ci95_high"]),
                    seed_values=tuple(seed_values),
                )
            )
        estimates[language] = rows
    return _source_cost_plot(estimates)


def cross_run_source_loss(source_costs: pd.DataFrame) -> Figure:
    """Plot source-language loss after rows are copied across training runs."""

    conditions = {
        "co_trained": ("reference", "Co-trained source rows\n(reference)"),
        "same_language_other_run": (
            "same_language",
            "Same-language rows\nfrom another run",
        ),
        "initial": ("initial", "Initial rows"),
        "other_language": ("other_language", "Other-language rows"),
    }
    estimates: dict[str, list[SourceCostEstimate]] = {}
    for language in SOURCE_LANGUAGES:
        language_rows = source_costs.loc[source_costs["source_language"].eq(language)]
        rows = []
        for condition_name in conditions:
            selected = language_rows.loc[language_rows["condition"].eq(condition_name)]
            seed_means = selected.groupby("seed")["source_cost_nll"].mean()
            if len(seed_means) != 12:
                raise ValueError(
                    f"expected 12 source-language seed means for {language}"
                )
            mean, low, high = _student_interval(seed_means.to_numpy(dtype=float))
            condition, label = conditions[condition_name]
            rows.append(
                SourceCostEstimate(
                    condition=condition,
                    label=label,
                    mean=mean,
                    low=low,
                    high=high,
                    seed_values=tuple(seed_means.to_numpy(dtype=float)),
                )
            )
        estimates[language] = rows
    return _source_cost_plot(estimates)


def cross_run_transfer_spelling_effects(
    components: pd.DataFrame,
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Compare co-trained rows with rows copied from other training runs."""

    contrast_ids = {
        "spanish": {
            "same": "spanish_same_language_other_run_benefit",
            "selectivity": "spanish_same_language_over_other_language",
            "exact_difference": "spanish_co_trained_minus_same_language_other_run",
        },
        "german": {
            "same": "german_same_language_other_run_benefit",
            "selectivity": "german_same_language_over_other_language",
            "exact_difference": "german_co_trained_minus_same_language_other_run",
        },
    }
    condition_labels = ("Co-trained", "Same language\n(other run)", "Other\nlanguage")
    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=(7.5, 4.85), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.55, 1.0))
        profile_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
        equivalence_ax = figure.add_subplot(grid[1, :])
        for panel_index, (ax, language) in enumerate(
            zip(profile_axes, SOURCE_LANGUAGES, strict=True)
        ):
            positions = np.arange(3, dtype=float)
            for outcome in ("step_zero", "learning_curve_auc"):
                style = OUTCOME_STYLE[outcome]
                same = _seed_values(
                    seed_estimates,
                    contrast_id=contrast_ids[language]["same"],
                    language=language,
                    outcome=outcome,
                )
                selectivity = _seed_values(
                    seed_estimates,
                    contrast_id=contrast_ids[language]["selectivity"],
                    language=language,
                    outcome=outcome,
                )
                exact_difference = _seed_values(
                    seed_estimates,
                    contrast_id=contrast_ids[language]["exact_difference"],
                    language=language,
                    outcome=outcome,
                )
                values = np.asarray(
                    [
                        (
                            same[seed] + exact_difference[seed],
                            same[seed],
                            same[seed] - selectivity[seed],
                        )
                        for seed in sorted(same)
                    ]
                )
                means = values.mean(axis=0)
                half_width = (
                    T_95_DF_11 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
                )
                x = positions + cast(float, style["offset"])
                color = str(LANGUAGE_STYLE[language]["color"])
                for point_position, seed_column in zip(x, values.T, strict=True):
                    _seed_points(
                        ax,
                        values=seed_column,
                        position=float(point_position),
                        color=color,
                        horizontal=False,
                    )
                ax.errorbar(
                    x,
                    means,
                    yerr=half_width,
                    color=color,
                    marker=str(style["marker"]),
                    markerfacecolor=color if bool(style["filled"]) else "white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    linestyle="none",
                    markersize=5.2,
                    capsize=2.4,
                    linewidth=1.25,
                    zorder=3,
                )
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xticks(positions, condition_labels)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.set_ylim(-0.04, 0.36)
            ax.set_yticks((0, 0.1, 0.2, 0.3))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        profile_axes[0].set_ylabel(
            "Change in spelling effect from initial rows\n(standardized units)"
        )
        profile_axes[1].tick_params(labelleft=False)

        equivalence_ax.axvspan(-0.05, 0.05, color=LIGHT_GRAY, zorder=0)
        for boundary in (-0.05, 0.05):
            equivalence_ax.axvline(
                boundary,
                color=INTERVAL_BOUNDARY,
                linestyle=(0, (2, 2)),
                linewidth=0.8,
            )
        positions = np.arange(4)[::-1]
        for position, (language, outcome) in zip(
            positions,
            (
                ("spanish", "step_zero"),
                ("spanish", "learning_curve_auc"),
                ("german", "step_zero"),
                ("german", "learning_curve_auc"),
            ),
            strict=True,
        ):
            row = _one_row(
                components,
                contrast_id=contrast_ids[language]["exact_difference"],
                language=language,
                outcome=outcome,
            )
            style = OUTCOME_STYLE[outcome]
            color = str(LANGUAGE_STYLE[language]["color"])
            seed_differences = _seed_values(
                seed_estimates,
                contrast_id=contrast_ids[language]["exact_difference"],
                language=language,
                outcome=outcome,
            )
            _seed_points(
                equivalence_ax,
                values=[seed_differences[seed] for seed in sorted(seed_differences)],
                position=float(position),
                color=color,
            )
            _interval_point(
                equivalence_ax,
                estimate=float(row["estimate"]),
                low=float(row["ci_low"]),
                high=float(row["ci_high"]),
                position=float(position),
                color=color,
                marker=str(style["marker"]),
                filled=bool(style["filled"]),
            )
        equivalence_ax.axvline(0, color=MUTED, linewidth=0.8)
        equivalence_ax.set_xlim(-0.06, 0.06)
        equivalence_ax.set_xticks((-0.05, 0, 0.05))
        equivalence_ax.set_yticks(
            positions,
            (
                "Spanish, before continuation",
                "Spanish, across continuation",
                "German, before continuation",
                "German, across continuation",
            ),
        )
        equivalence_ax.set_title(
            "(c) Co-trained minus same-language rows from another run"
        )
        equivalence_ax.set_xlabel("Difference in spelling effect (standardized units)")
        equivalence_ax.grid(axis="x")
        equivalence_ax.set_axisbelow(True)
        equivalence_ax.text(
            -0.048,
            0.96,
            "Equivalence range ±0.05",
            transform=equivalence_ax.get_xaxis_transform(),
            ha="left",
            va="top",
            color=MUTED,
            fontsize=7.7,
        )
        figure.legend(
            handles=_outcome_legend(),
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        return figure


def cross_run_checkpoint_profiles(checkpoint_profiles: pd.DataFrame) -> Figure:
    """Plot cross-run spelling effects over English continuation."""

    relation_styles = {
        "co_trained_rows": {
            "label": "Co-trained",
            "color": INK,
            "linestyle": "-",
            "marker": "o",
        },
        "other_run_same_language_donor": {
            "label": "Same language (other run)",
            "color": MATCH_COLOR,
            "linestyle": "--",
            "marker": "s",
        },
        "other_language_donor": {
            "label": "Other language",
            "color": MISMATCH_COLOR,
            "linestyle": ":",
            "marker": "^",
        },
    }
    checkpoints = sorted(
        int(value) for value in checkpoint_profiles["english_updates"].unique()
    )
    x = np.log1p(np.asarray(checkpoints, dtype=float))
    labeled_checkpoints = (0, 50, 200, 800, 6_100)
    marker_indices = [
        index
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint in labeled_checkpoints
    ]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.5, 3.2), sharey=True, constrained_layout=True
        )
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            for relation, style in relation_styles.items():
                selected = checkpoint_profiles.loc[
                    checkpoint_profiles["language"].eq(language)
                    & checkpoint_profiles["estimand"].eq(relation)
                ]
                grouped = selected.groupby("english_updates")["estimate"]
                means = grouped.mean().reindex(checkpoints).to_numpy(dtype=float)
                standard_deviation = (
                    grouped.std(ddof=1).reindex(checkpoints).to_numpy(dtype=float)
                )
                counts = grouped.count().reindex(checkpoints).to_numpy(dtype=float)
                if not np.all(counts == 12):
                    raise ValueError(
                        f"expected 12 checkpoint values for {language}/{relation}"
                    )
                half_width = (
                    T_SIMULTANEOUS_95_DF_11 * standard_deviation / np.sqrt(counts)
                )
                color = str(style["color"])
                ax.fill_between(
                    x,
                    means - half_width,
                    means + half_width,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    means,
                    color=color,
                    linestyle=str(style["linestyle"]),
                    marker=str(style["marker"]),
                    markevery=marker_indices,
                    markersize=2.8,
                    markeredgewidth=0.6,
                    linewidth=1.45,
                    label=str(style["label"]),
                )
            ax.axhline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.set_xticks(
                np.log1p(labeled_checkpoints),
                ("0", "50", "200", "800", "6,100"),
            )
            ax.set_xlabel("English updates (log(1 + updates) scale)")
            ax.set_ylim(-0.04, 0.36)
            ax.set_yticks((0, 0.1, 0.2, 0.3))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        axes[0].set_ylabel(
            "Change in spelling effect from initial rows\n(standardized units)"
        )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside lower center", ncol=3, frameon=False
        )
        return figure


def row_to_token_assignment_effects(
    components: pd.DataFrame,
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Show how much of the trained-row effect is lost and remains."""

    map_markers = {1: "o", 2: "s", 3: "^"}
    groups = (
        ("spanish", "step_zero", "Spanish, before continuation"),
        ("spanish", "learning_curve_auc", "Spanish, across continuation"),
        ("german", "step_zero", "German, before continuation"),
        ("german", "learning_curve_auc", "German, across continuation"),
    )
    language_block_gap = 0.45
    positions = np.asarray(
        (3.0 + language_block_gap, 2.0 + language_block_gap, 1.0, 0.0)
    )
    map_offsets = {1: 0.14, 2: 0.0, 3: -0.14}
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.5, 3.85),
            sharey=True,
            constrained_layout=True,
        )
        lost_ax, remaining_ax = axes
        remaining_ax.axvspan(-0.10, 0.10, color=LIGHT_GRAY, zorder=0)
        for boundary in (-0.10, 0.10):
            remaining_ax.axvline(
                boundary,
                color=INTERVAL_BOUNDARY,
                linestyle=(0, (2, 2)),
                linewidth=0.8,
            )
        for position, (language, outcome, _label) in zip(
            positions, groups, strict=True
        ):
            outcome_style = OUTCOME_STYLE[outcome]
            color = str(LANGUAGE_STYLE[language]["color"])
            for map_number, marker in map_markers.items():
                point_position = float(position + map_offsets[map_number])
                for ax, contrast_id in (
                    (
                        lost_ax,
                        f"correct_assignment_advantage__map_{map_number}",
                    ),
                    (
                        remaining_ax,
                        f"remaining_effect_after_reassignment__map_{map_number}",
                    ),
                ):
                    row = _one_row(
                        components,
                        contrast_id=contrast_id,
                        language=language,
                        outcome=outcome,
                    )
                    seeds = _seed_values(
                        seed_estimates,
                        contrast_id=contrast_id,
                        language=language,
                        outcome=outcome,
                    )
                    _seed_points(
                        ax,
                        values=[seeds[seed] for seed in sorted(seeds)],
                        position=point_position,
                        color=color,
                        half_span=0.023,
                        marker=marker,
                        size=10,
                        alpha=0.20,
                    )
                    _interval_point(
                        ax,
                        estimate=float(row["estimate"]),
                        low=float(row["ci_low"]),
                        high=float(row["ci_high"]),
                        position=point_position,
                        color=color,
                        marker=marker,
                        filled=bool(outcome_style["filled"]),
                        marker_size=6.0,
                    )
        for ax in axes:
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_yticks(positions, [label for _language, _outcome, label in groups])
            ax.set_ylim(-0.55, 3.8)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
        lost_ax.set_title("(a) Effect lost")
        lost_ax.set_xlabel("Effect lost\n(standardized units; 95% intervals)")
        lost_ax.set_xlim(-0.01, 0.27)
        lost_ax.set_xticks((0, 0.1, 0.2))
        remaining_ax.set_title("(b) Effect remaining")
        remaining_ax.set_xlabel("Effect remaining\n(standardized units; 90% intervals)")
        remaining_ax.set_xlim(-0.115, 0.115)
        remaining_ax.set_xticks((-0.10, -0.05, 0, 0.05, 0.10))
        remaining_ax.text(
            -0.108,
            -0.48,
            "Equivalence range ±0.10",
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=7.5,
        )
        map_handles = tuple(
            Line2D(
                [],
                [],
                color=MUTED,
                marker=marker,
                markersize=5.5,
                linestyle="none",
                label=f"Reassignment {map_number}",
            )
            for map_number, marker in map_markers.items()
        )
        figure.legend(
            handles=map_handles,
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        return figure


def row_to_token_assignment_persistence(checkpoint_estimates: pd.DataFrame) -> Figure:
    """Plot assignment effects as English continuation proceeds."""

    estimand_styles = {
        "trained_row_effect": {
            "label": "Trained-row spelling effect",
            "color": INK,
            "linestyle": "-",
            "marker": "o",
        },
        "correct_assignment_advantage": {
            "label": "Correct-assignment advantage",
            "color": MATCH_COLOR,
            "linestyle": "--",
            "marker": "s",
        },
        "remaining_effect_after_reassignment": {
            "label": "Effect remaining after reassignment",
            "color": MISMATCH_COLOR,
            "linestyle": ":",
            "marker": "^",
        },
    }
    checkpoints = sorted(
        int(value) for value in checkpoint_estimates["english_updates"].unique()
    )
    x = np.log1p(np.asarray(checkpoints, dtype=float))
    labeled_checkpoints = (0, 50, 200, 800, 6_100)
    marker_indices = [
        index
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint in labeled_checkpoints
    ]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.5, 3.2), sharey=True, constrained_layout=True
        )
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            for estimand, style in estimand_styles.items():
                selected = checkpoint_estimates.loc[
                    checkpoint_estimates["language"].eq(language)
                    & checkpoint_estimates["estimand"].eq(estimand)
                ].sort_values("english_updates")
                if list(selected["english_updates"].astype(int)) != checkpoints:
                    raise ValueError(
                        f"incomplete assignment trajectory for {language}/{estimand}"
                    )
                estimate = selected["estimate"].to_numpy(dtype=float)
                low = selected["simultaneous95_low"].to_numpy(dtype=float)
                high = selected["simultaneous95_high"].to_numpy(dtype=float)
                color = str(style["color"])
                ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
                ax.plot(
                    x,
                    estimate,
                    color=color,
                    linestyle=str(style["linestyle"]),
                    marker=str(style["marker"]),
                    markevery=marker_indices,
                    markersize=2.8,
                    markeredgewidth=0.6,
                    linewidth=1.45,
                    label=str(style["label"]),
                )
            ax.axhline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.set_xticks(
                np.log1p(labeled_checkpoints),
                ("0", "50", "200", "800", "6,100"),
            )
            ax.set_xlabel("English updates (log(1 + updates) scale)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Difference in spelling effect\n(standardized units)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside lower center", ncol=3, frameon=False
        )
        return figure


def larger_decoder_assignment_effects(
    components: pd.DataFrame,
    seed_estimates: pd.DataFrame,
) -> Figure:
    """Plot the larger-decoder row effect and assignment advantage."""

    labels = (
        "Trained-row spelling effect",
        "Correct assignment\nminus reassignment 1",
        "Correct assignment\nminus reassignment 2",
        "Correct assignment\nminus reassignment 3",
    )
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.5, 3.3), sharex=True, sharey=True, constrained_layout=True
        )
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            contrast_ids = (
                f"{language}_initial_minus_correctly_assigned",
                f"{language}_correct_assignment_advantage__map_1",
                f"{language}_correct_assignment_advantage__map_2",
                f"{language}_correct_assignment_advantage__map_3",
            )
            positions = np.arange(4)[::-1]
            for position, contrast_id in zip(positions, contrast_ids, strict=True):
                row = _one_row(
                    components,
                    contrast_id=contrast_id,
                    language=language,
                )
                color = INK if position == positions[0] else MATCH_COLOR
                seeds = _seed_values(
                    seed_estimates,
                    contrast_id=contrast_id,
                    language=language,
                    outcome="step_zero",
                    expected_count=14,
                )
                _seed_points(
                    ax,
                    values=[seeds[seed] for seed in sorted(seeds)],
                    position=float(position),
                    color=color,
                )
                _interval_point(
                    ax,
                    estimate=float(row["estimate"]),
                    low=float(row["ci_low"]),
                    high=float(row["ci_high"]),
                    position=float(position),
                    color=color,
                    marker="D" if position == positions[0] else "o",
                )
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_yticks(positions, labels)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.set_xlabel(
                "Difference in spelling effect\n"
                "(nats over the 0 to 1 similarity range)"
            )
        return figure


def independent_initialization_spelling_effects(
    condition_summaries: pd.DataFrame,
    pair_estimates: pd.DataFrame,
) -> Figure:
    """Plot effects before and across continuation for different initializations."""

    conditions = (
        "same_language_same_starting_weights",
        "same_language_different_starting_weights",
        "other_language_different_starting_weights",
    )
    condition_labels = (
        "Same language\nsame starting weights",
        "Same language\ndifferent starting weights",
        "Other language\ndifferent starting weights",
    )
    styles = {
        "same_language_same_starting_weights": CONDITION_STYLE["reference"],
        "same_language_different_starting_weights": CONDITION_STYLE["same_language"],
        "other_language_different_starting_weights": CONDITION_STYLE["other_language"],
    }
    outcome_rows = (
        (
            "step_zero",
            "before continuation",
        ),
        (
            "learning_curve_auc",
            "across continuation",
        ),
    )
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(7.5, 5.6),
            sharex="row",
            sharey=True,
            constrained_layout=True,
        )
        positions = np.arange(len(conditions))[::-1]
        for row_index, (outcome, outcome_title) in enumerate(outcome_rows):
            for column_index, language in enumerate(SOURCE_LANGUAGES):
                ax = axes[row_index, column_index]
                color = str(LANGUAGE_STYLE[language]["color"])
                ax.axvline(0, color=MUTED, linewidth=0.8)
                for position, condition in zip(positions, conditions, strict=True):
                    row = _one_row(
                        condition_summaries,
                        condition=condition,
                        language=language,
                        outcome=outcome,
                    )
                    style = styles[condition]
                    _seed_points(
                        ax,
                        values=_pair_values(
                            pair_estimates,
                            condition=condition,
                            language=language,
                            outcome=outcome,
                        ),
                        position=float(position),
                        color=color,
                        half_span=0.06,
                    )
                    estimate = float(row["estimate"])
                    _interval_point(
                        ax,
                        estimate=estimate,
                        low=float(row["ci_low"]),
                        high=float(row["ci_high"]),
                        position=float(position),
                        color=color,
                        marker=str(style["marker"]),
                    )
                    _mean_label(
                        ax,
                        estimate=estimate,
                        position=float(position),
                        color=color,
                    )
                ax.set_yticks(
                    positions,
                    condition_labels,
                )
                ax.set_ylim(-0.35, 2.35)
                ax.tick_params(axis="y", labelleft=column_index == 0)
                ax.grid(axis="x")
                ax.set_axisbelow(True)
                xlabel = (
                    "Spelling-effect change\n"
                    "(nats over the 0 to 1\n"
                    "similarity range)"
                    if outcome == "step_zero"
                    else "Spelling-effect change\n"
                    r"(nats $\times$ log-update over the"
                    "\n"
                    "0 to 1 similarity range)"
                )
                ax.set_xlabel(xlabel, fontsize=8.8)
                panel_letter = chr(97 + 2 * row_index + column_index)
                ax.set_title(
                    f"({panel_letter}) {LANGUAGE_LABEL[language]}, {outcome_title}"
                )
        return figure


def independent_initialization_word_surprisal(
    word_surprisal: pd.DataFrame,
) -> Figure:
    """Plot mean word surprisal immediately after row installation."""

    conditions = (
        "initial",
        "same_language_same_starting_weights",
        "same_language_different_starting_weights",
        "other_language_different_starting_weights",
    )
    labels = (
        "Initial rows",
        "Same language\nsame starting weights",
        "Same language\ndifferent starting weights",
        "Other language\ndifferent starting weights",
    )
    styles = {
        "initial": CONDITION_STYLE["initial"],
        "same_language_same_starting_weights": CONDITION_STYLE["reference"],
        "same_language_different_starting_weights": CONDITION_STYLE["same_language"],
        "other_language_different_starting_weights": CONDITION_STYLE["other_language"],
    }
    positions = np.arange(len(conditions))[::-1]
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.5, 3.45),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        for panel_index, (ax, language) in enumerate(
            zip(axes, SOURCE_LANGUAGES, strict=True)
        ):
            color = str(LANGUAGE_STYLE[language]["color"])
            initial_row = _one_row(
                word_surprisal,
                language=language,
                condition="initial",
            )
            ax.axvline(
                float(initial_row["mean"]),
                color=color,
                linestyle=(0, (1.5, 2.2)),
                linewidth=0.8,
                alpha=0.65,
            )
            for position, condition in zip(positions, conditions, strict=True):
                row = _one_row(word_surprisal, language=language, condition=condition)
                style = styles[condition]
                estimate = float(row["mean"])
                _interval_point(
                    ax,
                    estimate=estimate,
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                    position=float(position),
                    color=color,
                    marker=str(style["marker"]),
                )
                _mean_label(
                    ax,
                    estimate=estimate,
                    position=float(position),
                    color=color,
                )
            ax.set_yticks(positions, labels)
            ax.set_title(f"({chr(97 + panel_index)}) {BODY_PANEL_TITLE[language]}")
            ax.set_xlim(16.6, 31.5)
            ax.set_ylim(-0.35, len(conditions) - 0.45)
            ax.set_xticks((18, 21, 24, 27, 30))
            ax.grid(axis="x")
            ax.set_axisbelow(True)
        figure.supxlabel(
            "Mean English whole-word surprisal after row installation\n"
            "(nats per word, lower is better)",
            fontsize=9.5,
        )
        return figure
