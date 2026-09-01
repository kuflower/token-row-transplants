# Row-to-token assignment

This experiment kept the trained vectors fixed and changed which token IDs they
represented. Correctly assigned rows were compared with three fixed,
frequency-local reassignments in Spanish and German. The analysis uses twelve
paired seeds.

Correct assignment had a positive advantage for every language, outcome, and
reassignment. The spelling effect left after reassignment was inside the ±0.10
equivalence range in the primary analysis, but that conclusion was less stable
under some alternative outcome definitions. The three reassignments are fixed
interventions, not independent samples of possible mappings.

| File | What it contains |
|---|---|
| [`claims.csv`](claims.csv) | Decisions and Holm-adjusted p values for the two main claims |
| [`components.csv`](components.csv) | Correct-assignment and remaining-effect estimates |
| [`seed_estimates.csv`](seed_estimates.csv) | Paired-seed estimates behind the components |
| [`assignment_checkpoints.csv`](assignment_checkpoints.csv) | Trained-row, correct-assignment, and remaining effects during continuation |
| [`target_piece_count_sensitivity.csv`](target_piece_count_sensitivity.csv) | Component estimates after adding target piece count |
| [`outcome_sensitivity.csv`](outcome_sensitivity.csv) | Compact comparison of five outcome specifications |
| [`mapping_sensitivity.csv`](mapping_sensitivity.csv) | Compact comparison of translation-selection and token-sequence features |
| [`word_bootstrap_summary.csv`](word_bootstrap_summary.csv) | Additional paired word-bootstrap bounds |

The sensitivity files contain aggregate estimates only. They do not include
individual words or translation-table rows.
