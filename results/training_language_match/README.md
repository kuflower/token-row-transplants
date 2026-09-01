# Training-language match

This experiment crossed Spanish-, German-, and English-trained receiving bodies
with initial, Spanish-trained, German-trained, and English-trained rows. All
comparisons use the same twelve paired seeds.

Spanish and German rows produced their largest spelling effects in a body trained
on the same language. The fully English-trained model had the lowest final English
perplexity. English-trained rows improved the final English performance of the
Spanish- and German-trained bodies, but did not make them match the English-trained
model. The cross-body equivalence result applies to the standardized whole-word
outcome; the per-piece analysis is reported separately.

| File | What it contains |
|---|---|
| [`claims.csv`](claims.csv) | Decisions and Holm-adjusted p values for the six main claims |
| [`components.csv`](components.csv) | Component means and confidence intervals |
| [`seed_estimates.csv`](seed_estimates.csv) | Paired-seed estimates behind the components |
| [`performance.csv`](performance.csv) | English NLL, final perplexity, and source-language NLL for all twelve body-row combinations |
| [`performance_seed_values.csv`](performance_seed_values.csv) | Seed values shown in the final English perplexity and source-language loss plots |
| [`english_nll_checkpoints.csv`](english_nll_checkpoints.csv) | English-row minus matched-source-row NLL during continuation |
| [`spelling_checkpoints.csv`](spelling_checkpoints.csv) | Spelling-effect trajectories during continuation |
| [`spelling_sensitivity.csv`](spelling_sensitivity.csv) | Target-piece-adjusted, per-piece, and raw-scale spelling estimates |

Positive values in `english_nll_checkpoints.csv` mean that English-trained rows
have higher NLL than matched source-trained rows. In the spelling tables, positive
contrasts mean that the trained source rows have the stronger spelling effect.
