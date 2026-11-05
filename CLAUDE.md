# Rules for changing Shravan (for people and for AI assistants)

Read `ARCHITECTURE.md` first, then `shravan/README.md` for the module you change and `tests/README.md` for the tests.
This file is the short list of rules; those files say why.

This repository is public. Anything committed can be read by anyone, so the first rules are about what never goes in.

## What never goes into git
- No audio of any kind (recordings, clips, MP3s). Recordings live in iCloud Drive (`config.yaml`, `audio:`), the
  clips cut from them in `work/`; `.gitignore` keeps both out. Never weaken those lines.
- No names. `recordings.csv` names people by initials only, and so do results, logs and commit messages.
- No model weights or checkpoints (`*.safetensors`, `*.bin`, `*.onnx`, `checkpoints/`, `models/`): they are large and
  can be rebuilt. A finished model is released into the Nitya Upasana site, not here.
- No credentials or tokens, and no paths that reveal more than `~/` and the iCloud Drive folder name.

## The texts
- `texts/` is a copy of the site's `content/texts/`, made only by `python scripts/sync_texts.py`, with `texts/SOURCE`
  naming the site commit. Never edit a file in `texts/`: correct the text on the site, then sync.
- What the model learns to write is the plain Devanagari of each line (`shravan/labels.py`): no svara marks, verse
  numbers, kampa digits or dandas. A change to those rules is a change to every model: say so in the commit, and
  retrain the baseline.

## Recordings and fair tests
- Every recording has one row in `recordings.csv` before it is used.
- Recordings marked `heldout=yes`, and the held-out hymns in `config.yaml`, are never trained on. They are what the
  test scores mean. `tests/` checks the split.
- A model is judged only on held-out audio, and always against the untrained model on the same audio.

## Runs
- Every training or test run writes one file to `results/`, named by date, model and run, with: the texts commit
  (`texts/SOURCE`), the settings used, the recordings used, the seed and the scores. A score without that file is
  not a result.
- Settings change in `config.yaml`, not inside the scripts. A script reads the config; it never hard-codes a path,
  model name or number that belongs there.
- Runs are on the M4 Mac with 16 GB (`mps`), full precision (`fp32`): half precision is unreliable for training there.

## Code
- Python 3.12. The packages are pinned in `requirements.txt` (`requirements-dev.txt` for the checks); a new package
  or version is pinned there in the same change, and `pip check` must pass.
- One module per topic in `shravan/`. Logic that needs no audio, model or GPU (parsing, labels, matching, splitting)
  is a plain function there and is unit-tested in `tests/`. The commands in `scripts/` stay thin: they read the
  config and call the modules.
- Type hints everywhere; `mypy` must pass. Layout and lint by ruff (`ruff format`, `ruff check`): double quotes,
  4-space indent, lines up to 120 characters.
- Comments say why, in plain English. No stray `print` in the modules; the scripts report what they did.
- Every change comes with a test, new or updated. Before a pull request: `ruff format --check . && ruff check . &&
  mypy && pytest` (the same as CI, `.github/workflows/tests.yml`).
- When a module or a command changes, update `shravan/README.md` or `README.md` in the same change.

## Branches and releases
- Changes go on a branch and are merged by pull request; `main` is what the next run trains from.
- A model released to the site is named `chant-vN` (N counts up) and its results file is committed with it. The site's
  own rules (its `CLAUDE.md`) govern how the model files are added there.
