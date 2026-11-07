# The modules

One module per topic. Logic that needs no audio, model or GPU is a plain function, unit-tested in `tests/`. The
commands in `scripts/` read the config and call these.

| Module | What it does | Tested in |
|---|---|---|
| `config.py` | Reads `config.yaml` into a typed `Config` (paths resolved from the repository, `~` expanded) | `test_config.py` |
| `texts.py` | Reads the site's text files from `texts/`: header, blocks, line pairs (original, everyday English). Names the line of any mistake, as the site does | `test_texts.py` |
| `labels.py` | `plain(line)`: what the model learns to write, the plain Devanagari words of an original line (no svara marks, verse numbers, kampa digits, dandas, avagraha, joiners or hyphens) | `test_labels.py` |
| `units.py` | A text's units: Pavamana's numbered verses with their Rigveda reference and hymn (`verses()`), any other text's blocks keyed `132` or `once-7` (`stanzas()`; a block too long for one clip in pieces of a few lines, `4.1`, `4.2`, `pieces()`), a text in the order it is chanted, an aarti's dhrupad after every stanza (`sung()`), and the blocks a recording names, in its order (`select()`) | `test_pipeline.py` |
| `recordings.py` | The rows of `recordings.csv` (each text with the blocks chanted, `parts`), and each recording's safe name for its working files (`slug`) | `test_pipeline.py` |
| `align.py` | Forced alignment of a long recording against its verses with torchaudio's MMS aligner, window by window, with a wildcard at each end; each verse's start, end and score | `test_pipeline.py` (the window maths; the Mac runs it) |
| `split.py` | Which split a verse belongs to (train, validation, test), by hymn | `test_config.py` |
| `data.py` | The clips made by `scripts/cut.py`, from every recording's manifest | — |
| `hear.py` | What a model writes for pieces of audio, a batch at a time: on the GPU, or a model shrunk for the phone (an ONNX folder) with ONNX Runtime on the CPU; a CTC model (wav2vec2) letter by letter, each piece on its own | — |
| `ctc.py` | The CTC model's letters: a label as it writes it (Devanagari, words joined by `|`, `text()`), its alphabet with our missing letters added (`alphabet()`), labels to ids (`encode()`) and its 20 ms choices back to words (`decode()`) | `test_ctc.py` |
| `augment.py` | Training clips changed a little each time (speed, volume, noise), for both kinds of model | — |
| `follow.py` | Where the chanting is, from what was heard: the end of its best fuzzy match in the text (`where()`), searched whole or only near the last place (`nearby()`) | `test_pipeline.py` |
| `metrics.py` | Letters wrong (character error rate) and verses found (is the nearest verse the right one), what the model wrote cleaned by the label rules first | `test_pipeline.py` |
| `results.py` | One file per run in `results/`: texts commit, recordings, split, settings and scores | — |
| `env.py` | Points Python at certifi's certificates (python.org's Python on a Mac has none, and model downloads fail); caps PyTorch's GPU memory (`gpu_memory`) so a run fails clearly instead of going into swap | `test_config.py` (the cap) |
