# Row-to-token assignment in a larger decoder

The assignment intervention was repeated at step zero in a tied decoder with
97,835,520 active parameters. Fourteen seeds used the same tokenizer, data,
training schedule, and three fixed reassignments as the main assignment study.

Trained rows produced a larger spelling effect than initial rows, and correctly
assigned rows produced a larger effect than all three reassignments in both languages.
This is one additional decoder configuration, not a scaling study. Several
architecture choices changed together, so differences in effect size cannot be
assigned to one change.

| File | What it contains |
|---|---|
| [`claims.csv`](claims.csv) | Decisions and Holm-adjusted p values for the two main claims |
| [`components.csv`](components.csv) | Trained-row and correct-assignment estimates |
| [`seed_estimates.csv`](seed_estimates.csv) | Seed-level estimates behind the components |

These estimates use raw step-zero surprisal slopes and should not be compared
numerically with the standardized estimates from the main assignment study.
