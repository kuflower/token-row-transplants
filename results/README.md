# Results

The directories cover the four main experiments and the larger-decoder check. Each
one contains the claim-level decisions, component estimates, and the seed- or
pair-level values behind them. Smaller checkpoint and sensitivity tables cover the
additional analyses shown in the appendix and notebook.

| Experiment | Question | Directory |
|---|---|---|
| Training-language match | Does transfer depend on the training languages of the rows and receiving body? | [`training_language_match/`](training_language_match/) |
| Transfer across training runs | Do rows transfer between runs that share starting weights? | [`cross_run_transfer/`](cross_run_transfer/) |
| Row-to-token assignment | Does transfer depend on keeping each trained vector with its token ID? | [`row_to_token_assignment/`](row_to_token_assignment/) |
| Independent initialization | What changes when the rows and body begin from different weights? | [`independent_initialization/`](independent_initialization/) |
| Larger decoder | Does the assignment result appear in one larger decoder configuration? | [`larger_decoder_assignment/`](larger_decoder_assignment/) |

[`robustness/`](robustness/) contains the raw-scale estimates for the cross-run
and assignment experiments. The repository does not include corpora, model
checkpoints, raw probe records, or training traces.
