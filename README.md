# Shravan (श्रवण)

*Shravanam*, listening, is the first of the nine forms of bhakti. This project teaches a small speech model to listen to
our family chanting the Nitya Upasana texts, so the app can follow along and keep the place in the text.

This repository is **public for now**, while the work is getting to a good state. It holds the training code, the list
of recordings and the test results. It never holds the audio itself, and `.gitignore` keeps audio and model files out.
Because anyone can read it, `recordings.csv` names people by initials only.

Before changing anything, read `CLAUDE.md` (the rules) and `ARCHITECTURE.md` (how the pieces fit).

## Where things live

| What | Where | Why |
|---|---|---|
| Training and test code, settings, results | this repository | so every run can be repeated and compared |
| The audio | iCloud Drive → `development/shravan-training/`, never in git | private, too big for git, and synced from the phone |
| Clips cut from the audio | `work/`, never in git | private like the audio; can be made again from it |
| The texts | `texts/`, copied from the site by `python scripts/sync_texts.py` | a fixed copy per run; `texts/SOURCE` names the site commit |
| The finished, shrunk model | copied into the site as `models/chant-vN/` | the site never depends on this repository |

## The layout

```
shravan-model/
  CLAUDE.md             the rules for changing this repository (.cursorrules points Cursor to it)
  ARCHITECTURE.md       how the pipeline fits together, from texts and recordings to a model on the phone
  config.yaml           every setting: paths, model, language, held-out hymns, training, targets
  recordings.csv        one row per recording: file, date, texts chanted, who chanted (initials), held out or not
  shravan/              the code, one module per topic (see shravan/README.md)
  scripts/              the commands you run; thin, they read the config and call shravan/
  tests/                logic tests, no audio or GPU needed (see tests/README.md)
  texts/                the 51 chant texts and library.json, exactly as on the site
  results/              one file per run: the texts commit, settings, recordings used, scores
  requirements.txt      the training packages, pinned; requirements-dev.txt the checks
  pyproject.toml        settings for ruff, mypy and pytest
```

## Setting up the Mac (once)

```sh
brew install ffmpeg
cd ~/development/shravan-model
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -c "import torch; print(torch.backends.mps.is_available())"   # should print True
```

## The commands

Run from `shravan-model/` with the environment active (`source .venv/bin/activate`). The first time,
`pip install -e .` makes the `shravan` package importable from `scripts/`.

| Command | What it does |
|---|---|
| `python scripts/sync_texts.py` | Copies the texts from `../nitya-upasana` into `texts/` and records the site commit |
| `python scripts/align.py "<recording>"` | Converts the recording to 16 kHz mono and finds where each verse starts and ends (`work/align/`) |
| `python scripts/cut.py "<recording>"` | Cuts one clip per verse with its label and split (`work/clips/`); verses aligned badly are left out |
| `python scripts/evaluate.py <model> [--split validation]` | Letters wrong and verses found on the test (or validation) clips; writes `results/` |
| `python scripts/follow.py <model>` | The app's own measure: plays each tested recording as a phone hears it and scores how often the right place is found 2, 3 and 5 seconds into a verse, searching the whole text and following |
| `python scripts/train.py <model> [--resume] [--name N]` | Fine-tunes Whisper on the training clips (a family recording's counted `family_repeat` times), keeping the best checkpoint on validation (`checkpoints/N`); `--resume` carries on a stopped run. Base takes about 85 minutes on the M4 |
| `ruff format . && ruff check . && mypy && pytest` | The checks, the same as CI; run them before a pull request |

`<recording>` is its file name in `recordings.csv`; `<model>` is `openai/whisper-base`, `openai/whisper-small` or a
trained one, `checkpoints/openai-whisper-base/best`.

## Where we are

Trained on one published recording of Pavamana Suktam (60 minutes of verses), tested on hymns 61–63, which training
never saw (`results/`):

| Model | Letters wrong | Verses found | Test time for 10 minutes of chant |
|---|---|---|---|
| Whisper base, untrained | 226% | 0 of 79 | — |
| Whisper small, untrained | 177% | 0 of 79 | — |
| **Whisper base, trained** | **10.0%** | **79 of 79** | 12 s |
| Whisper small, trained | 9.9% | 79 of 79 | 30 s |

Base is the model to take forward: as good as small, with a third of the size.

A family recording of a text never trained on (SJ, Lalita Sahasranama, stanzas 132–183 and the Dhyanam, 8 minutes,
held out) and the app's own measure, the right place within 3 seconds of a verse starting (`scripts/follow.py`):

| Trained Whisper base | Letters wrong | Verses found | Right place within 3 s |
|---|---|---|---|
| Pavamana Suktam, hymns 61–63 | 10% | 100% of 610 | 100% |
| Lalita Sahasranama, SJ (never trained on) | 45% | 96% of 190 | 96% |

Trained on one published recording of one text, the model already follows a family voice through another. Next: more
family recordings to test on, then the phone-sized model (ONNX, int8).

## Adding recordings

1. Put the recordings (Voice Memos, MP3s) in iCloud Drive → `development/shravan-training/`. On the Mac, right-click the
   folder and choose **Keep Downloaded**, so iCloud never swaps a recording out for a placeholder during training.
2. Add one row per recording to `recordings.csv`, with initials in the `who` column. If the session followed the usual
   routine, `routine` is enough in the `texts` column. A recording of part of a text, or with a part said again, names
   its blocks after a colon, in the order chanted: `vedic-chant-lalita-sahasranama: 132..183 once-7 once-2..once-5`
   (a stanza by its number, anything else by its kind and count: the seventh `[once]` is `once-7`). Without parts, a
   text is taken whole as it is chanted, an aarti's dhrupad after every stanza. A recording of several texts lists
   them in order, separated by `;`.
3. Leave the held-out day's rows marked `heldout=yes`. They are never trained on, so the scores mean something.

## The targets

A model is released to the site only when, on the held-out recordings:

- fewer than 15% of Devanagari letters are wrong (character error rate), and
- the app lands on the right line within 3 seconds of starting it more than 95% of the time.

When a text is added or corrected on the site, run `python scripts/sync_texts.py` and commit `texts/`. Each run
records the site commit its texts came from (`texts/SOURCE`), so a score can always be traced to the words it was
tested on.

The plan in full: [Training your own chanting model on the M4 Mac](https://claude.ai/artifact/X2AFmBBm3uuY1TXc71oUg3).
