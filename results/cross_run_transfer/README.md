# Transfer across training runs

This experiment copied rows between source-language runs that began from the same
weights but used different training batches. The receiving body, tokenizer, and
evaluation data were held fixed within each of the twelve paired seeds.

Same-language rows from another run produced a clear spelling effect and a larger one
than other-language rows. Their effect was close to the co-trained condition under the
±0.05 equivalence rule. This experiment does not test copying between independently
initialized models.

| File | What it contains |
|---|---|
| [`claims.csv`](claims.csv) | Decisions and Holm-adjusted p values for the three main claims |
| [`components.csv`](components.csv) | Component means and confidence intervals |
| [`seed_estimates.csv`](seed_estimates.csv) | Paired-seed estimates behind the components |
| [`checkpoint_profiles.csv`](checkpoint_profiles.csv) | Co-trained, same-language cross-run, and other-language spelling profiles |
| [`english_performance.csv`](english_performance.csv) | Final English NLL and perplexity for each seed and row condition |
| [`source_language_costs.csv`](source_language_costs.csv) | Source-language NLL before and after English continuation |
| [`source_language_contrasts.csv`](source_language_contrasts.csv) | Paired contrasts in source-language NLL change |
| [`spelling_sensitivity.csv`](spelling_sensitivity.csv) | Target-piece-count-adjusted spelling estimates |

`source_language_costs.csv` averages the two transfer directions within each seed.
Spanish and German use different held-out corpora. Compare row conditions within a
language, not the absolute NLL values across languages.
