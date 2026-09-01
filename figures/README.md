# Figures

This directory contains the experiment overview and the twelve analysis figures used
by the notebook. Each figure is committed as PDF for the report and PNG for the
notebook and GitHub.

The analysis figures are built from the tables under [`results/`](../results/) by
[`render_analysis_figures.py`](../scripts/render_analysis_figures.py). The overview is
a static diagram shared with the report.

Faint points show individual seeds or initialization pairs when those estimates are
included in the result tables. Larger markers show means and intervals. The three
checkpoint plots use mean lines and simultaneous 95% bands instead. The
word-surprisal table contains condition summaries rather than its twelve pair values,
so that figure shows means and 95% intervals without faint points. Numbers beside
markers are the plotted means.

| Study | Files | What it shows |
|---|---|---|
| Overview | [`PNG`](token_row_transplant_overview.png) · [`PDF`](token_row_transplant_overview.pdf) | Parent training, row installation, and English continuation |
| Training-language match | [`PNG`](training_language_match_spelling_effects.png) · [`PDF`](training_language_match_spelling_effects.pdf) | Spelling effects when row and body languages match or differ |
| Training-language match | [`PNG`](training_language_match_final_perplexity.png) · [`PDF`](training_language_match_final_perplexity.pdf) | Final English perplexity and the fully English-trained reference |
| Training-language match | [`PNG`](training_language_match_checkpoint_trajectories.png) · [`PDF`](training_language_match_checkpoint_trajectories.pdf) | English NLL and spelling-effect changes during continuation |
| Training-language match | [`PNG`](training_language_match_source_loss.png) · [`PDF`](training_language_match_source_loss.pdf) | Source-language loss after restoring the original rows |
| Transfer across training runs | [`PNG`](cross_run_transfer_spelling_effects.png) · [`PDF`](cross_run_transfer_spelling_effects.pdf) | Co-trained, same-language cross-run, and other-language rows |
| Transfer across training runs | [`PNG`](cross_run_transfer_checkpoint_profiles.png) · [`PDF`](cross_run_transfer_checkpoint_profiles.pdf) | Spelling-effect profiles during English continuation |
| Transfer across training runs | [`PNG`](cross_run_transfer_source_loss.png) · [`PDF`](cross_run_transfer_source_loss.pdf) | Source-language loss after cross-run transplantation |
| Row-to-token assignment | [`PNG`](row_to_token_assignment_effects.png) · [`PDF`](row_to_token_assignment_effects.pdf) | The spelling effect lost and remaining after reassignment |
| Row-to-token assignment | [`PNG`](row_to_token_assignment_persistence.png) · [`PDF`](row_to_token_assignment_persistence.pdf) | Assignment effects during English continuation |
| Larger-decoder check | [`PNG`](larger_decoder_assignment_effects.png) · [`PDF`](larger_decoder_assignment_effects.pdf) | Step-zero trained-row and correct-assignment effects |
| Independent initialization | [`PNG`](independent_initialization_spelling_effects.png) · [`PDF`](independent_initialization_spelling_effects.pdf) | Spelling effects with the same or different starting weights |
| Independent initialization | [`PNG`](independent_initialization_word_surprisal.png) · [`PDF`](independent_initialization_word_surprisal.pdf) | Mean whole-word surprisal immediately after row installation |

When Spanish and German share an axis, amber denotes Spanish and blue denotes German.
Within separate language panels, color distinguishes the row conditions being compared.
