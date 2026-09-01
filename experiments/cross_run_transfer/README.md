# Transfer across training runs

## Question

Does the spelling effect transfer across runs that have the same starting weights?

## Design

For each of 12 seeds, one saved initialization branches into Spanish A, Spanish B,
German A, and German B parent runs with different deterministic batch streams. Each
of the four trained bodies receives initial rows and trained rows from all four
parents. The grid contains 20 assemblies per seed and 240 English continuations.

The seed-level analysis distinguishes co-trained rows, rows from the other
same-language run, and rows from the two other-language runs. The main comparisons
test the spelling effect from same-language rows from another run, their advantage
over other-language rows, and whether their mean difference from co-trained rows lies
within ±0.05. Spanish and German must pass on both the step-zero and learning-curve
outcomes. The seed is the inferential unit.

## Result

Rows from another same-language run reproduced nearly all of the co-trained spelling
effect and produced a larger effect than other-language rows. Co-trained rows had a
slightly larger mean effect, but all four co-trained-minus-cross-run intervals were
inside ±0.05.

Final English NLL differed little between co-trained and same-language cross-run rows.
After the original source rows were restored, source NLL rose slightly less for the
co-trained condition and much more for initial and other-language rows. These are
descriptive comparisons.

## Scope

This tests transfer across training runs whose models have the same starting weights.
It does not establish direct transfer across independently initialized models. The
equivalence result concerns the population mean on the standardized outcomes, not
equality in every seed. The other-language condition is a control; the experiment did
not test whether its spelling effect is zero.

The result is limited to Spanish and German source training, English continuation,
one fixed tokenizer, and the tied decoder family used here.

## Check the analysis

[`experiment.toml`](experiment.toml) summarizes the model and comparison grid.
Check the result tables and figures from the repository root:

```bash
make verify-results
```

The [result tables](../../results/cross_run_transfer/) contain the included estimates,
checkpoint profiles, and endpoint analyses. Recomputing them from model outputs
requires the original checkpoints.
