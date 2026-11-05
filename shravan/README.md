# The modules

One module per topic. Logic that needs no audio, model or GPU is a plain function, unit-tested in `tests/`. The
commands in `scripts/` read the config and call these.

| Module | What it does | Tested in |
|---|---|---|
| `config.py` | Reads `config.yaml` into a typed `Config` (paths resolved from the repository, `~` expanded) | `test_config.py` |
| `texts.py` | Reads the site's text files from `texts/`: header, blocks, line pairs (original, everyday English). Names the line of any mistake, as the site does | `test_texts.py` |
| `labels.py` | `plain(line)`: what the model learns to write, the plain Devanagari words of an original line (no svara marks, verse numbers, kampa digits, dandas, avagraha or joiners) | `test_labels.py` |
| `units.py` | A text's numbered verses (`verses()`), each with its Rigveda reference, hymn and label: what is aligned, cut and split | `test_pipeline.py` |
| `recordings.py` | The rows of `recordings.csv`, and each recording's safe name for its working files (`slug`) | `test_pipeline.py` |
| `align.py` | Forced alignment of a long recording against its verses with torchaudio's MMS aligner, window by window, with a wildcard at each end; each verse's start, end and score | `test_pipeline.py` (the window maths; the Mac runs it) |
| `split.py` | Which split a verse belongs to (train, validation, test), by hymn | `test_config.py` |
| `data.py` | The clips made by `scripts/cut.py`, from every recording's manifest | — |
| `metrics.py` | Letters wrong (character error rate) and verses found (is the nearest verse the right one), what the model wrote cleaned by the label rules first | `test_pipeline.py` |
| `results.py` | One file per run in `results/`: texts commit, recordings, split, settings and scores | — |
| `env.py` | Points Python at certifi's certificates (python.org's Python on a Mac has none, and model downloads fail); caps PyTorch's GPU memory (`gpu_memory`) so a run fails clearly instead of going into swap | `test_config.py` (the cap) |
