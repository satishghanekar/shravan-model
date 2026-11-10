# Shravan (श्रवण)

*Shravanam*, listening, is the first of the nine forms of bhakti. This project teaches a small speech model to listen to
our family chanting the Nitya Upasana texts, so the app can follow along and keep the place in the text.

This repository is **public for now**, while the work is getting to a good state. It holds the training code, the list
of recordings and the test results. It never holds the audio itself, and `.gitignore` keeps audio and model files out.
Because anyone can read it, `recordings.csv` names people by initials only.

Before changing anything, read `CLAUDE.md` (the rules) and `ARCHITECTURE.md` (how the pieces fit).

## The current model

**chant-v4** (on the site since 2026-11-09; both forms come from `checkpoints/ctc-pro-all`): Vakyansh's Sanskrit
wav2vec2 fine-tuned on every recording not held out. An 8-bit form runs on a phone's CPU (103 MB, `chant-v4`); a 4-bit
form on its GPU (90 MB, `chant-v4-q4`), three times quicker on an iPhone.

On the recordings it never trained on, 3 seconds into each unit, on the CPU:

| Recording | Letters wrong | Right place, following | Right place, after a jump |
|---|---|---|---|
| Pavamana, 1st published | 13.4% | 100% | 96.2% |
| Pavamana, Challakere | 6.3% | 100% | 95.6% |
| SG's Purusha, 2nd take | 17.6% | 100% | 82.6% |
| SG's Purusha, 1st take | 15.3% | 100% | 92.0% |
| Lalita, one voice | 10.3% | 100% | 99.5% |
| SJ's Lalita, a group | 36.7% | 98.3% | 48.3% |

No score here covers an aarti, an ashtak or Marathi: those recordings are all trained on. The GPU's numbers are within
a point or so of these; the runs behind them, and every model before, are in [`results/`](results/README.md).

To build it again:

```sh
python scripts/train_ctc.py --name ctc-pro-all                                 # about 2 hours on the M4
python scripts/export.py checkpoints/ctc-pro-all/best models/ctc-v3-int8       # (the 8-bit form's folder keeps this name)
python scripts/export.py checkpoints/ctc-pro-all/best models/ctc-v4-q4 --q4
python scripts/to_site.py models/ctc-v3-int8 models/site/chant-v4
python scripts/to_site.py models/ctc-v4-q4 models/site/chant-v4-q4
```

**Licences.** The starting model's [Hugging Face page](https://huggingface.co/Harveenchadha/vakyansh-wav2vec2-sanskrit-sam-60)
names no licence (checked 2026-11-10); the Vakyansh project's code and model lists on GitHub
([`Open-Speech-EkStep`](https://github.com/Open-Speech-EkStep)) are MIT. The site's weights are fine-tuned partly on
published recordings, which are not ours and are never in this repository.

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
  results/              one file per run: the texts commit, settings, recordings used, scores; its README the history
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
| `python scripts/evaluate.py <model> [--split validation]` | Letters wrong and verses found on the test (or validation) clips, a clip's verse looked for among every text its recording chants; writes `results/` |
| `python scripts/follow.py <model>` | The app's own measure, by Listen's own method (`shravan/follow.py`, its settings in `config.yaml`, `listen:`): plays each tested recording as Listen hears it (the last 4 s) and scores how often the right place is found 2, 3 and 5 seconds into a unit, following (the window lined up with the text near Listen's mark) and lost (every text searched). |
| `python scripts/export.py <model> <folder> [--q4]` | Shrinks a trained model for the phone: ONNX, int8, about 108 MB (`models/`, never in git); `--q4`, its matrix weights in 4 bits for a phone's GPU (90 MB); test the folder like any model |
| `python scripts/train_ctc.py [<model>] [--resume] [--name N] [--layers L] [--only published\|family]` | Fine-tunes the model (`model` in `config.yaml` unless another is named: wav2vec2, a letter for every 20 ms) on the training clips (a family recording's counted `family_repeat` times), our letters it lacks added to its alphabet, keeping the best checkpoint on validation (`checkpoints/N`). `--resume` carries on a stopped run, `--layers` keeps the first L layers, `--only` trains on one kind of recording, `--set key=value` changes one `training:` setting for a comparison. Up to about 4.5 hours on the M4 |
| `python scripts/speed.py <folder>` | Times a phone model on this Mac's CPU with 4 threads, as an iPhone runs it, on 2, 4 and 5 s of sound. Only the ratio between models means much on the phone |
| `ruff format . && ruff check . && mypy && pytest` | The checks, the same as CI; run them before a pull request |

`<recording>` is its file name in `recordings.csv`; `<model>` is the pretrained model's name (`model` in `config.yaml`) or a
trained one, `checkpoints/ctc-pro-all/best`.

## Adding recordings

1. Put the recordings (Voice Memos, MP3s) in iCloud Drive → `development/shravan-training/`. On the Mac, right-click the
   folder and choose **Keep Downloaded**, so iCloud never swaps a recording out for a placeholder during training.
2. Add one row per recording to `recordings.csv`, with initials in the `who` column. A family recording is listed by its
   file name alone, without the folder it is in (named after a person): the file is found in any folder. A recording of part of a text, or with a part said again, names
   its blocks after a colon, in the order chanted: `vedic-chant-lalita-sahasranama: 132..183 once-7 once-2..once-5`
   (a stanza by its number, anything else by its kind and count: the seventh `[once]` is `once-7`). Without parts, a
   text is taken whole as it is chanted, an aarti's dhrupad after every stanza. A recording of several texts lists
   them in order, separated by `;`.
3. Leave the held-out day's rows marked `heldout=yes`. They are never trained on, so the scores mean something.

## The targets

A model is released to the site only when, on the held-out recordings:

- fewer than 15% of Devanagari letters are wrong (character error rate), and
- the app lands on the right line within 3 seconds of starting it more than 95% of the time.

The second is measured by `scripts/follow.py` (following, 3 s), by Listen's own method since 2026-11-10: the last 4 s
lined up with the text near the mark. The earlier "right place" figures in `results/README.md` were measured the
earlier way (the last 6 s, the letters read and fuzzy-searched), so an older model is compared again before a release.

When a text is added or corrected on the site, run `python scripts/sync_texts.py` and commit `texts/`. Each run
records the site commit its texts came from (`texts/SOURCE`), so a score can always be traced to the words it was
tested on.

The plan in full: [Training your own chanting model on the M4 Mac](https://claude.ai/artifact/X2AFmBBm3uuY1TXc71oUg3).
