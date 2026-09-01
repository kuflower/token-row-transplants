# Method

The experiments ask when a trained token matrix remains compatible with a receiving
transformer body. They hold the tokenizer, token IDs, English continuation
stream, probe, and evaluation procedure fixed while varying where the rows and the
rest of the model came from. This is a controlled parameter-transplant design, not a
general comparison of multilingual language models.

## Experimental objects

| Term | Meaning in this repository |
|---|---|
| Token row | The learned vector assigned to one token ID. The studied decoders tie the input embedding and output projection, so installing a row changes both the embedding lookup and output logits. |
| Token matrix | The complete tied matrix installed for an experimental condition. Its vectors are token rows, and each transplant moves the full matrix. |
| Body | Every trained model parameter other than the token rows installed at the intervention boundary. |
| Training run | One optimization trajectory. Separate runs may begin from the same initial weights but follow distinct language-training trajectories. |
| Token assignment | The mapping from trained row vectors to token IDs. Reassignment moves the same vectors without retraining them. |
| Model initialization | One random draw of a model's initial parameters. Models with the same initialization begin from identical weights. |
| Initialization pair | Two independently initialized model families paired for Experiment 4. The initialization pair is its inferential unit. |

At the intervention boundary, the selected donor rows are installed into the same
receiving checkpoint for the paired arms. The English optimizer state and
seed-indexed English batch stream are reset consistently. Row-source comparisons
hold the receiving checkpoint fixed; body-language and initialization contrasts
vary it by design.

## Composite English probe

The probe combines target words from the six train and development files in the
British Council's BEA 2026 Knowledge-based Vocabulary Lists (KVLs) with prefixes
from a held-out English Wikipedia stream.

Probe construction lowercases and trims the KVL English targets, retains single ASCII
alphabetic words, and scans Wikipedia in source order for at most eight mid-sentence
occurrences per target. Sentence-initial targets are excluded. At evaluation, each
score uses the final 96 BPE pieces before the target word.

Spelling similarity is normalized Levenshtein similarity after lowercasing and
removing diacritics. When a KVL item lists more than one Spanish or German form, the
analysis uses the form with the highest similarity to the English target. For repeated
English spellings, it uses the first-listed KVL item. Model tokenization keeps the
original case and diacritics.

The KVL processing step yields 6,234 eligible target words. The complete probe
contains 48,358 contexts for 6,205 of them. The scientific analysis keeps the 6,023
words with at least five contexts.

The English slice and generated probe are not included in this repository. Their
role in the analysis and the relevant source licenses are documented in
[Reproducibility](REPRODUCIBILITY.md) and [`DATA_LICENSE.md`](../DATA_LICENSE.md).

## Outcomes

Each probe word has Spanish-English and German-English spelling-similarity scores,
frequency, length, and part-of-speech covariates. Every experimental condition fits
the same joint regression with both similarity scores. The training-language-match
experiment retains both slopes in every condition. The other experiments use the
slope matching the receiving body's source language for their main claims. Spanish
and German similarity always appear together, so neither score is silently omitted.

`s0` is whole-word summed surprisal immediately after row installation. The
learning-curve area (AUC) integrates, over `log(1 + step)`, the nonnegative gap
between each word's running minimum surprisal and its terminal running minimum. This
best-so-far summary does not measure later deterioration. Experiments 1 through 3
standardize the per-word outcome within each analysis panel before regression. The
independent-initialization experiment uses the raw (unstandardized) outcomes, and the
larger-decoder check uses only raw step-zero surprisal. Results on these scales are not
interchangeable.

For trained-versus-initial comparisons, a positive spelling-slope contrast means that
trained rows produced a stronger spelling effect. Other contrasts follow the named
subtraction in the result tables. A spelling-slope contrast is not by itself a claim
about mean English loss, semantic retention, downstream task quality, or a language
model's general usefulness.

## Inference and decision rules

The seed or initialization pair is the inferential unit. Probe words, model arms,
checkpoints, and fixed reassignment maps are not independent replications.
Multi-component claims use intersection-union logic, so every component must pass.
Holm correction controls the claim family. An equivalence test shows that an interval
lies inside its specified margin on the analysis scale. It does not establish
equality or an exact zero effect.

Directional components use 95% confidence intervals. Equivalence components use
90% confidence intervals against their ±0.10 or ±0.05 margins.
Seed points and checkpoint profiles in the figures are descriptive unless a caption
identifies them as part of a main claim.

## Scope

The main studies use Spanish and German training, English continuation, one tokenizer
and corpus family, tied input and output rows, and a small decoder architecture. The
larger-decoder check repeats the row-to-token assignment experiment in one fixed larger
decoder. Several architectural dimensions change together, so it is not a randomized
scaling test. The three reassignment maps are fixed perturbations, not a population
sample of possible maps. The independent-initialization experiment tests unchanged
copying only. It does not estimate the effect of fitting an adapter or coordinate
map.
