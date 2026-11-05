# The modules

One module per topic. Logic that needs no audio, model or GPU is a plain function, unit-tested in `tests/`. The
commands in `scripts/` read the config and call these.

| Module | What it does | Tested in |
|---|---|---|
| `config.py` | Reads `config.yaml` into a typed `Config` (paths resolved from the repository, `~` expanded) | `tests/test_config.py` |
| `texts.py` | Reads the site's text files from `texts/`: header, blocks, line pairs (original, everyday English). Names the line of any mistake, as the site does | `tests/test_texts.py` |
| `labels.py` | `plain(line)`: what the model learns to write, the plain Devanagari words of an original line (no svara marks, verse numbers, kampa digits, dandas, avagraha or joiners) | `tests/test_labels.py` |

To come with the pipeline (`ARCHITECTURE.md`): alignment, clips, the train/held-out split, testing, training and
export.
