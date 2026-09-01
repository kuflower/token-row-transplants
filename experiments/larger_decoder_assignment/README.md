# Row-to-token assignment in a larger decoder

## Question

Do the trained-row spelling effect and correct-assignment effect replicate at step
zero in a larger tied-decoder configuration?

## Design

The replication uses a decoder with width 768, 12 layers, 12 attention heads, and
97,835,520 active parameters. The tokenizer, corpora, training exposure, and three
fixed frequency-local reassignments match the main assignment study.

The study uses 14 seeds. The two reciprocal receiving-model directions are averaged
within each seed. The main raw step-zero outcome tests whether correctly paired
trained rows produce a larger spelling effect than initial rows and each of the three
reassignments on both language axes. The study does not run an English continuation.

## Result

Both claims were supported. The trained-row spelling effect was +0.931 for Spanish
and +1.038 for German. Correct-assignment advantages ranged from +0.339 to +0.502
across the six language-map components. Every 95% interval was
above zero.

Both Holm-adjusted p values were below .001.

## Scope

This is a replication at one fixed decoder geometry, not a randomized comparison
between model sizes. It does not identify a capacity effect, model-size interaction,
or scaling law. The outcome is a raw whole-word surprisal slope measured before
English continuation, so the study does not test persistence during continued
training.

The three mappings remain fixed interventions, not samples from a population of
assignments. Magnitudes on the raw scale should not be compared directly with the
standardized coefficients in the main assignment experiment.

## Check the analysis

[`experiment.toml`](experiment.toml) summarizes the model, training, and comparison
settings. Run the result and figure checks:

```bash
make verify-results
```

The [result tables](../../results/larger_decoder_assignment/) contain the main
and descriptive estimates. Model checkpoints and raw probe trajectories are not
included.
