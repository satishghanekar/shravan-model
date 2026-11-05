# Shravan (श्रवण)

*Shravanam*, listening, is the first of the nine forms of bhakti. This project teaches a small speech model to listen to
our family chanting the Nitya Upasana texts, so the app can follow along and keep the place in the text.

This repository is **private**. It holds the training code, the list of recordings and the test results. It never holds
the audio itself.

## Where things live

| What | Where | Why |
|---|---|---|
| Training and test scripts, settings, results | this repository | so every run can be repeated and compared |
| The audio | `~/chant-audio/` (backed up to iCloud or a drive), never in git | private, and far too big for git |
| The texts | read from `../nitya-upasana/content/texts/` | one copy of the words; a new text is picked up by the next run |
| The finished, shrunk model | copied into the site as `models/chant-vN/` | the site never depends on this repository |

## The layout

```
shravan/
  config.yaml       which texts, which model, training settings, the targets
  recordings.csv    one row per memo: file, date, texts chanted, who chanted, held out or not
  requirements.txt  the Python packages
  results/          one file of test scores per run
  train.py          (to come) trains Whisper base or small on the recordings
  test.py           (to come) scores a model on the held-out recordings
  align.py          (to come) finds where each line starts in a recording, using the known text
```

## Setting up the Mac (once, about 20 minutes)

```sh
brew install python@3.11 ffmpeg
cd ~/development/shravan
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # should print True
```

## Adding recordings

1. Copy the Voice Memos into `~/chant-audio/`.
2. Add one row per memo to `recordings.csv`. If the session followed the usual routine, `routine` is enough in the
   `texts` column.
3. Leave the held-out day's rows marked `heldout=yes`. They are never trained on, so the scores mean something.

## The targets

A model is released to the site only when, on the held-out recordings:

- fewer than 15% of Devanagari letters are wrong (character error rate), and
- the app lands on the right line within 3 seconds of starting it more than 95% of the time.

Each run records the site commit its texts came from, so a score can always be traced to the words it was tested on.

The plan in full: [Training your own chanting model on the M4 Mac](https://claude.ai/artifact/X2AFmBBm3uuY1TXc71oUg3).
