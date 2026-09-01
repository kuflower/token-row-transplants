# Training-language match

## Question

How does the training-language match between installed token rows and a receiving
body affect the spelling effect? How do the assembled models compare with an
equal-update English performance reference?

## Design

For each of 12 seeds, one initialization forks into Spanish-, German-, and
English-trained parents. After 6,100 parent updates, every trained body receives four
row sources from the same seed: initial, Spanish-trained, German-trained, and
English-trained rows. The 3 × 4 grid gives 12 assemblies per seed, 36 parent
trajectories, and 144 English continuations.

The reciprocal word analysis uses the six Spanish and German body-row cells. It
measures step-zero whole-word surprisal and the learning-curve summary, then estimates
Spanish-English and German-English spelling effects on a within-run standardized
scale. The English-performance analysis uses the full grid and reports held-out
English negative log-likelihood before and during continuation. The seed is the
inferential unit. All six main claims share one Holm correction family.

## Result

Matched Spanish and German rows produced their strongest spelling effect with bodies
trained on the same language. The effect was larger in the matching body, and each
source-trained body favored rows trained on its language. All four cross-body
intervals were inside the ±0.10 margin on the standardized whole-word outcomes.

The fully English-trained body and rows had the lowest held-out English loss under
the same update budget. Neither an English body with source-trained rows nor a
source-trained body with English rows reached that reference. Installing English
rows on source-trained bodies did not reduce the step-zero and
log-step-weighted NLL gaps.

Checkpoint analyses found that English rows overtook matched source rows by update
50 and finished slightly ahead on English text. English rows gave a small final
English advantage but a much larger source-language cost after the original rows were
restored. Those checkpoint and restoration comparisons are exploratory or
descriptive, not additional main claims.

## Scope

The English-trained model is a performance reference for this experiment only. Equal
update counts do not make the Spanish, German, and English corpora linguistically
equivalent. The cross-body equivalence result applies to the population mean on the
main standardized whole-word outcomes; it does not establish a zero effect or
interchangeable components.

The tokenizer and token IDs remain fixed, and the rows are tied between input and
output roles. Evaluation after restoring the original source rows measures their fit
to the body, not retained semantic knowledge.

## Check the analysis

[`experiment.toml`](experiment.toml) summarizes the model and comparison grid.
Check the result tables and figures from the repository root:

```bash
make verify-results
```

The [result tables](../../results/training_language_match/) contain the main
estimates and exploratory follow-ups. Replaying training requires the corpora and
token streams used by the experiment.
