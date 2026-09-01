# Row-to-token assignment

## Question

Does the spelling effect remain when the training-time row-to-token assignment is
changed?

## Design

The experiment reuses the parent panel from the cross-run study. For each
of 12 seeds, Spanish and German rows from one run are installed in the other
same-language body. Each receiving model is evaluated with initial rows, correctly
paired cross-run rows, and three reassignments of the same set of trained vectors.
The reassignments were chosen before outcomes were examined and applied to every
seed. The two reciprocal receiving-model directions are averaged within seed.

Each reassignment uses every vector exactly once and pairs every vector with a
different token. Tokens are ranked by source-training frequency, split into adjacent
blocks of at most 33, shuffled within each block, and cyclically shifted. The
receiving body, donor run, tokenizer, token IDs, English batches, and complete vector
set remain fixed. The experiment contains 240 English continuations.

The main claims test whether correct assignment produces a larger spelling effect than
every reassignment and whether the spelling effect remaining after reassignment lies
within ±0.10. Both claims must hold for Spanish and German, both primary outcomes,
and all three fixed reassignments.

## Result

Correct assignment produced a larger spelling effect than every reassignment. The
advantages ranged from +0.146 to +0.246, and every 95% interval was above zero.
Estimates of the spelling effect remaining after reassignment ranged from +0.028 to
+0.092, and every 90% interval was inside ±0.10 on the main whole-word outcomes.

English continuation reduced the assignment effect sharply, but the advantage of
correct pairing remained detectable at update 6,100 in the exploratory checkpoint
analysis. Sensitivity analyses retained the correct-assignment direction, while the
size of the residual depended on the outcome specification.

## Scope

This intervention changes row-to-token assignment, not geometric alignment. Because
the matrix is tied, reassignment changes both how a token is read and how it is
scored. The experiment cannot separate those two roles.

The three fixed reassignments are correlated interventions rather than independent
draws from a population of permutations. The equivalence decision applies only to the
main standardized whole-word outcomes. It does not mean the residual is
exactly zero, that every seed lies inside the margin, or that alternate outcome
definitions inherit the same decision.

## Check the analysis

[`experiment.toml`](experiment.toml) summarizes the model and comparison grid.
Check the result tables and figures from the repository root:

```bash
make verify-results
```

The [result tables](../../results/row_to_token_assignment/) contain the main
estimates and sensitivity checks.
