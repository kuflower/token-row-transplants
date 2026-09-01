# Reproducing the analysis

This repository includes the experiment configurations, compact result tables,
plotting code, figures, and an executed notebook. The training corpora, vocabulary
probe, token streams, and model checkpoints are not included because of their
licenses, contents, or size.

## Environment

Run commands from the repository root with GNU Make and the exact Python version in
`.python-version`:

```bash
make environment
source .venv/bin/activate
```

`make environment` creates `.venv`, installs the versions in
`requirements-ci.txt`, installs the local package, and checks the environment. If
the default `PYTHON_BOOTSTRAP` resolves to a different patch release, pass the
matching interpreter explicitly.

## Results and figures

The main entry point is
[`results_overview.ipynb`](../notebooks/results_overview.ipynb). It reads the CSV
files under [`results/`](../results/) directly. It places the larger-decoder check
beside the row-to-token assignment experiment that it extends.

```bash
make verify-results       # check the figures and execute a fresh notebook copy
make notebook-refresh     # rewrite figures and stored notebook output
make check                # add tests, lint, formatting, and type checks
```

The notebook and render script call the same plotting functions. The twelve analysis
figures are committed as PDF for the report and PNG for the notebook and GitHub.
Their constructors live in [`plots.py`](../src/token_row_transplants/plots.py), and
[`render_analysis_figures.py`](../scripts/render_analysis_figures.py) provides the
non-notebook entry point. The experiment-overview diagram is a static figure. The
[figure index](../figures/README.md) lists every plot and its source experiment.

## Inputs not included

| Input | Role in the study | Source |
|---|---|---|
| Spanish, German, and English Wikipedia text | Parent training, English continuation, and held-out evaluation | `wikimedia/wikipedia`, `20231101` snapshots |
| Shared byte-level BPE tokenizer and token streams | Fix token identities and sampled training sequences | Trained from the Wikipedia inputs |
| English vocabulary probe | Measure whole-word surprisal and spelling similarity | British Council BEA 2026 Knowledge-based Vocabulary Lists paired with held-out English contexts |
| Model checkpoints | Supply receiving bodies and donor token rows | Produced by the parent-training runs described in the experiment configurations |

The Wikipedia and KVL sources and their licenses are listed in
[`DATA_LICENSE.md`](../DATA_LICENSE.md). The repository intentionally contains no
word-level KVL rows or Wikipedia text.

The configurations under [`experiments/`](../experiments/) summarize the model size,
conditions, seeds, update counts, and main comparisons. Re-running model
training requires obtaining the source data and rebuilding the derived inputs
locally. The included result tables are sufficient to inspect the reported estimates
and reproduce the analysis figures without those files.

## Probe data

[Method](METHOD.md#composite-english-probe) describes the sampling and filtering
procedure used for the held-out English contexts and KVL-selected probe. The
generated contexts and word-level feature table are not redistributed here. The
included analysis tables contain aggregate estimates rather than source prose or
KVL entries.
