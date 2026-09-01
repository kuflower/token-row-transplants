# Independent initialization

## Question

Does the same-language effect survive when rows are copied unchanged between models
that start from different random weights?

## Design

The study forms 12 pairs of independently initialized model families. Each family
contains Spanish A, Spanish B, German A, and German B parent runs. The paired
families use the same A and B source batch sequences and the same English
continuation sequence. A receiving model is evaluated with its initial rows,
same-language rows from another run in its own family, and same- or other-language
rows from the paired family. The receiving body is held fixed across conditions.

Rows are copied to the same token IDs without alignment, an adapter, or a change to
the row-to-token assignment. No additional training occurs before the immediate
evaluation. Within each pair, the analysis averages the A and B receiving runs and
both transfer directions between the two initializations. The initialization pair
is the inferential unit.

The main analysis uses raw spelling-similarity slopes rather than the
standardized outcomes used by the first three experiments. It compares
same-language rows from the paired family with initial rows and with other-language
rows from that family, at step zero and across English continuation. The
same-language condition from the receiving body's own family provides descriptive
context.

## Result

Neither joint claim was established. With different starting weights,
the step-zero same-language spelling-slope change was +0.370 for Spanish and +0.057
for German, down from +1.214 and +1.286 when the rows and body began from the same
weights. The German step-zero interval crossed zero, leaving the four-part
criterion unmet. The Holm-adjusted claim-level p value was .348.

Against other-language rows, all four point estimates went opposite the tested
direction, and the adjusted claim-level p value exceeded .999. That pattern is
descriptive and is not treated as a supported reverse claim.

An exploratory check compared mean whole-word surprisal immediately after row
installation. Relative to initial rows, different-start same-language rows raised
mean surprisal by 8.22 nats in Spanish-trained receivers and 6.52 nats in
German-trained receivers. Other-language rows from the paired initialization also
had high mean surprisal. The fitted spelling slope therefore should not be read as
average prediction quality.

## Scope

The result concerns direct unchanged copying. Across independently initialized
models, unchanged copying is a baseline rather than a complete adaptation method.
The experiment does not test an alignment, adapter, brief interface-training phase,
or another composition rule. Failure to establish the joint claims is not evidence
of zero transfer.

The outcomes are raw spelling-conditioned coefficients, not general English loss or
semantic capability. Their units differ between step zero and the learning-curve
summary, so magnitudes are interpreted within each outcome.

## Check the analysis

[`experiment.toml`](experiment.toml) summarizes the model, training, and comparison
settings. Run the result and figure checks:

```bash
make verify-results
```

The [result tables](../../results/independent_initialization/) include the main
components, initialization-pair estimates, and descriptive comparisons. Checkpoints
and raw probe trajectories are not included.
