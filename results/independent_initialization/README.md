# Independent initialization

This experiment paired twelve independently initialized model families. Each
receiving body was evaluated with initial rows, same-language rows that shared its
starting weights, and same- or other-language rows from the paired family. Rows
were copied to the same token IDs without an alignment step.

The two main joint claims were not established. The German step-zero interval for the
same-language effect included zero, and every same-language versus other-language
estimate pointed opposite to the tested direction. Mean whole-word surprisal was high
for both conditions copied across different starting weights, so the steeper spelling
slope should not be read as better average prediction.

| File | What it contains |
|---|---|
| [`claims.csv`](claims.csv) | Decisions and Holm-adjusted p values for the two main claims |
| [`components.csv`](components.csv) | Component means and confidence intervals |
| [`pair_estimates.csv`](pair_estimates.csv) | Initialization-pair estimates behind the components |
| [`condition_effects.csv`](condition_effects.csv) | Spelling effects for the three trained-row conditions |
| [`condition_pair_estimates.csv`](condition_pair_estimates.csv) | Initialization-pair values shown in the three-condition spelling plots |
| [`starting_weights_comparison.csv`](starting_weights_comparison.csv) | Same-language effects with the same and different starting weights |
| [`checkpoint_selectivity.csv`](checkpoint_selectivity.csv) | Same- versus other-language slope contrast at each saved checkpoint |
| [`word_surprisal.csv`](word_surprisal.csv) | Condition means and 95% intervals for whole-word surprisal immediately after row installation |

`checkpoint_selectivity.csv` contains aggregate checkpoint estimates. It has no
individual-word or translation-table records.
