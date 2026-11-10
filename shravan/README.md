# The modules

One module per topic. Logic that needs no audio, model or GPU is a plain function, unit-tested in `tests/`. The
commands in `scripts/` read the config and call these.

| Module | What it does | Tested in |
|---|---|---|
| `config.py` | Reads `config.yaml` into a typed `Config` (paths resolved from the repository, `~` expanded) | `test_config.py` |
| `texts.py` | Reads the site's text files from `texts/`: header, blocks, line pairs (original, everyday English). Names the line of any mistake, as the site does; a line after `[refrain]` in its block is marked as one | `test_texts.py` |
| `labels.py` | `plain(line)`: what the model learns to write, the plain Devanagari words of an original line (no svara marks, verse numbers, kampa digits, dandas, avagraha, joiners or hyphens) | `test_labels.py` |
| `units.py` | A text's units: Pavamana's numbered verses with their Rigveda reference and hymn (`verses()`), any other text's blocks keyed `132` or `once-7` (`stanzas()`; a block too long for one clip in pieces of a few lines, `4.1`, `4.2`, `pieces()`), a text in the order it is chanted, an aarti's dhrupad after every stanza (`sung()`), the blocks a recording names, in its order (`select()`), a recording's units across all its texts as chanted (`chanted()`) and every unit of its texts once (`candidates()`), and each block's key (`keyed()`) | `test_pipeline.py` |
| `recordings.py` | The rows of `recordings.csv` (each text with the blocks chanted, `parts`), each recording's safe name for its working files (`slug`), and where its audio is (`locate()`: a family recording listed by its file name alone, found in any folder) | `test_pipeline.py` |
| `align.py` | Forced alignment of a long recording against its verses with torchaudio's MMS aligner, window by window, with a wildcard at each end; each verse's start, end and score | `test_pipeline.py` (the window maths; the Mac runs it) |
| `split.py` | Which split a verse belongs to (train, validation, test), by hymn | `test_config.py` |
| `data.py` | The clips made by `scripts/cut.py`, from every recording's manifest; the training clips weighed (`weighed()`: a family recording's counted `family_repeat` times, or only one kind of recording); where a verse's clip starts and ends (`edges()`) | `test_pipeline.py` |
| `hear.py` | What a model hears, on the GPU or, shrunk for the phone (an ONNX folder), with ONNX Runtime on the CPU: the model loaded once (`Ctc`), its scores for each 20 ms (what `follow.py` lines up) and its letters, each piece on its own | `test_pipeline.py` |
| `ctc.py` | The model's letters: a label as it writes it (Devanagari, words joined by `|`, `text()`), its alphabet with our missing letters added (`alphabet()`), labels to ids (`encode()`), its 20 ms choices back to words (`decode()`), and which output is its blank (`blank_of()`) | `test_ctc.py` |
| `augment.py` | Training clips changed a little each time (speed, volume, noise) | `test_pipeline.py` |
| `follow.py` | Where the chanting is, found exactly as Listen finds it (the site's `js/core/align.js` and `js/core/follow.js`, line for line): a window's scores lined up with the text near the mark (`align()`), or its letters looked for (`locate()`), one hearing as Listen takes it (`hear()`); a recording's texts laid out as the site shows them (`book()`), where Listen's mark is while it follows (`mark_at()`), and whether a place found is the unit chanted (`right()`) | `test_follow.py`, against `tests/fixtures/listen.json` |
| `sound.py` | `normalise()`: sound brought to mean 0 and spread 1, as the model was trained and as the site does it; the one copy, used by training, testing and packing | `test_sound.py` |
| `package.py` | A model packed for the site: files cut into pieces under Cloudflare's 25 MiB and joined back with their SHA-256 (`split()`, `join()`), the check sound (`tone()`) and what the site must get from it (`check_of()`) | `test_pipeline.py` |
| `metrics.py` | Letters wrong (character error rate) and verses found (is the nearest verse the right one), what the model wrote cleaned by the label rules first | `test_pipeline.py` |
| `results.py` | One file per run in `results/`: texts commit, this repository's commit, recordings, split, settings and scores | `test_pipeline.py` |
| `env.py` | Points Python at certifi's certificates (python.org's Python on a Mac has none, and model downloads fail); caps PyTorch's GPU memory (`gpu_memory`) so a run fails clearly instead of going into swap; which device to run on (`device()`) | `test_config.py` (the cap) |
| `gpu.py` | `FreeMemory`, for training: hands the GPU's cached memory back every 50 steps and after each evaluation, or a run on clips of many lengths fills its cap | `test_pipeline.py` |
