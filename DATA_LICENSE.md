# Third-party data, attribution, and redistribution status

The project code and documentation are available under the [MIT License](LICENSE).
Upstream materials used by the experiments are available under separate terms. The
third-party and source-derived artifacts listed below are not included.

## Artifact status

| Artifact family | Upstream material | Relevant upstream terms | Included here | Handling |
|---|---|---|---|---|
| Spanish, German, and English corpora and derived token streams | Text from `wikimedia/wikipedia` | CC BY-SA 3.0 and GFDL | No | Obtain the source text from upstream and preserve its attribution and terms. |
| Shared byte-level BPE tokenizer | Trained from the Wikipedia corpora | Wikipedia source terms | No | Rebuild locally from the source corpora. |
| KVL source files and derived spelling features | Knowledge-based Vocabulary Lists from BEA 2026 Shared Task 1 | CC BY-NC 4.0 | No | Obtain the source files from the British Council repository and rebuild the features locally. |
| Composite English probe | KVL-selected target words paired with English Wikipedia prefixes | Both source licenses apply | No | Build and store locally; this repository does not redistribute the words or contexts. |

The repository includes project-authored code, configurations, aggregate result
tables, and figures. Those files are covered by `LICENSE` unless a file says
otherwise. The included tables do not contain Wikipedia prose or KVL entries.

## Upstream attribution

- **Wikipedia:** text by Wikipedia contributors, obtained through the
  [`wikimedia/wikipedia` dataset](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa)
  at revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.
  [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and GFDL.
- **Knowledge-based Vocabulary Lists / BEA 2026 Shared Task 1:** British Council,
  [repository revision `71010c7`](https://github.com/britishcouncil/bea2026st/tree/71010c7209f6b00ff29017a810fade9d78fe9578).
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Data handling

The boundary above is deliberately conservative. It records what this project
includes; it is not a general determination about what an upstream
license permits. Acquiring an input or running a builder does not change that
input's terms. Users who obtain the upstream material are responsible for following
the applicable license and attribution requirements.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the scientific role of
each input and the steps for reproducing the included analysis.
