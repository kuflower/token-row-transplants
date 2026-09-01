from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Any

from scripts.normalize_notebook import normalize_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/results_overview.ipynb"

ANALYSIS_FIGURE_BUILDERS = {
    "training_language_match_spelling_effects",
    "training_language_match_final_perplexity",
    "training_language_match_checkpoint_trajectories",
    "training_language_match_source_loss",
    "cross_run_transfer_spelling_effects",
    "cross_run_checkpoint_profiles",
    "cross_run_source_loss",
    "row_to_token_assignment_effects",
    "row_to_token_assignment_persistence",
    "larger_decoder_assignment_effects",
    "independent_initialization_spelling_effects",
    "independent_initialization_word_surprisal",
}
STATIC_NOTEBOOK_IMAGE = "token_row_transplant_overview.png"


def _notebook() -> dict[str, Any]:
    value: Any = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _code_tree(cell: dict[str, Any]) -> ast.Module:
    assert cell.get("cell_type") == "code"
    return ast.parse("".join(cell.get("source", [])))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _called_names(cell: dict[str, Any]) -> set[str]:
    return {
        name
        for node in ast.walk(_code_tree(cell))
        if isinstance(node, ast.Call)
        if (name := _call_name(node)) is not None
    }


def _string_literals(cell: dict[str, Any]) -> set[str]:
    return {
        node.value
        for node in ast.walk(_code_tree(cell))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _has_png_output(cell: dict[str, Any]) -> bool:
    return any(
        "image/png" in output.get("data", {}) for output in cell.get("outputs", [])
    )


def _html_tables(notebook: dict[str, Any]) -> list[str]:
    return [
        "".join(html)
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if (html := output.get("data", {}).get("text/html"))
        if "<table" in "".join(html)
    ]


def _table_captions(notebook: dict[str, Any]) -> list[str]:
    captions: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        calls = sorted(
            (
                node
                for node in ast.walk(_code_tree(cell))
                if isinstance(node, ast.Call) and _call_name(node) == "display_table"
            ),
            key=lambda call: (call.lineno, call.col_offset),
        )
        for call in calls:
            caption_keyword = next(
                keyword for keyword in call.keywords if keyword.arg == "caption"
            )
            assert isinstance(caption_keyword.value, ast.Constant)
            assert isinstance(caption_keyword.value.value, str)
            captions.append(caption_keyword.value.value)
    return captions


def test_results_notebook_has_clean_stored_outputs() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]

    assert code_cells
    assert all(isinstance(cell.get("execution_count"), int) for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert all("execution" not in cell.get("metadata", {}) for cell in code_cells)
    assert not re.search(r" at 0x[0-9a-fA-F]+", json.dumps(notebook))


def test_results_notebook_table_titles_render_as_header_rows() -> None:
    notebook = _notebook()
    tables = _html_tables(notebook)
    captions = _table_captions(notebook)

    assert tables
    assert len(tables) == len(captions)
    for caption, html in zip(captions, tables, strict=True):
        assert "<caption" not in html
        assert html.count("<table") == 1
        assert html.count('class="col_heading level0 col0"') == 1
        assert 'class="col_heading level1 col0"' in html
        assert html.count(f">{caption}</th>") == 1
        assert html.index('class="col_heading level0 col0"') < html.index(
            'class="col_heading level1 col0"'
        )


def test_analysis_plots_are_built_in_the_notebook() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    plot_cells: dict[str, dict[str, Any]] = {}
    all_calls: set[str] = set()
    for cell in code_cells:
        calls = _called_names(cell)
        all_calls.update(calls)
        for builder in calls & ANALYSIS_FIGURE_BUILDERS:
            assert builder not in plot_cells, f"{builder} is called more than once"
            plot_cells[builder] = cell

    assert set(plot_cells) == ANALYSIS_FIGURE_BUILDERS
    assert all(_has_png_output(cell) for cell in plot_cells.values())
    assert "save_figure" in all_calls


def test_notebook_only_embeds_the_experiment_overview_as_a_static_image() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    png_paths = {
        literal
        for cell in code_cells
        for literal in _string_literals(cell)
        if literal.endswith(".png")
    }

    assert {Path(path).name for path in png_paths} == {STATIC_NOTEBOOK_IMAGE}
    assert (REPOSITORY_ROOT / "figures" / STATIC_NOTEBOOK_IMAGE).is_file()


def test_results_notebook_uses_portable_paths() -> None:
    assert "/Users/" not in json.dumps(_notebook(), ensure_ascii=False)


def test_results_notebook_reads_every_included_result_table() -> None:
    notebook = _notebook()
    calls: set[tuple[str, str]] = set()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for node in ast.walk(_code_tree(cell)):
            if not isinstance(node, ast.Call) or _call_name(node) != "read_study_table":
                continue
            assert len(node.args) >= 2
            study, filename = node.args[:2]
            assert isinstance(study, ast.Constant) and isinstance(study.value, str)
            assert isinstance(filename, ast.Constant) and isinstance(
                filename.value, str
            )
            calls.add((study.value, filename.value))

    results_root = REPOSITORY_ROOT / "results"
    included_tables = {
        path.relative_to(results_root)
        for path in results_root.rglob("*.csv")
        if "runs-root" not in path.parts
    }
    notebook_tables = {Path(study) / filename for study, filename in calls}

    assert notebook_tables == included_tables


def test_notebook_normalization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "notebook.ipynb"
    shutil.copy2(NOTEBOOK_PATH, path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    cell["metadata"] = {
        "execution": {"iopub.status.busy": "timestamp"},
        "trusted": True,
    }
    cell["outputs"] = [
        {
            "output_type": "display_data",
            "metadata": {},
            "data": {"text/html": ["<p>result</p>"], "text/plain": ["at 0x123"]},
        }
    ]
    path.write_text(json.dumps(notebook), encoding="utf-8")

    normalize_notebook(path)
    first = path.read_bytes()
    normalize_notebook(path)
    normalized = json.loads(path.read_text(encoding="utf-8"))
    normalized_cell = next(
        cell for cell in normalized["cells"] if cell["cell_type"] == "code"
    )

    assert path.read_bytes() == first
    assert normalized_cell["metadata"] == {}
    assert "text/plain" not in normalized_cell["outputs"][0]["data"]
