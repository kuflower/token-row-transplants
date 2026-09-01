# Experiments

These directories describe the four main experiments and the larger-decoder
assignment check. The Python code is under
[`src/token_row_transplants/`](../src/token_row_transplants/), and the result tables
are under [`results/`](../results/).

## Studies

| Study | Directory | Question |
|---|---|---|
| 1. Training-language match | [`training_language_match/`](training_language_match/) | How does the training-language match between rows and body affect the spelling effect? |
| 2. Transfer across training runs | [`cross_run_transfer/`](cross_run_transfer/) | Does the spelling effect transfer across runs that share initial weights? |
| 3. Row-to-token assignment | [`row_to_token_assignment/`](row_to_token_assignment/) | Does the effect remain when the training-time row-to-token assignment is changed? |
| 4. Independent initialization | [`independent_initialization/`](independent_initialization/) | Does the same-language effect survive when rows are copied unchanged between models that start from different random weights? |
| Row-to-token assignment in a larger decoder | [`larger_decoder_assignment/`](larger_decoder_assignment/) | Do the trained-row and assignment effects replicate at step zero in a larger decoder configuration? |

The confidence intervals and equivalence ranges apply only to the outcomes and
comparisons defined for each experiment.

## Inspect and verify

Each study directory includes a compact summary of its model and comparison grid.
Run the included analysis with:

```bash
make verify-results
```

Repeating model training requires corpora, token streams, and checkpoints that are
not included here. The configurations, compact results, and analysis notebook can
be inspected from a clean checkout.
