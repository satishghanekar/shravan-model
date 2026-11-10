# Tests

The logic tests: everything that needs no audio, model or GPU, so they run in seconds on any machine and in CI
(`.github/workflows/tests.yml`).

## Running

```sh
source .venv/bin/activate
ruff format . && ruff check . && mypy && pytest
```

`pytest -k labels` runs one file's tests.

## What they cover

| File | Checks |
|---|---|
| `test_texts.py` | Every copied text parses; `texts/SOURCE` names a clean site commit; Pavamana Suktam has its 610 verses in order; a mistake in a text is named with its line |
| `test_labels.py` | The label rules on the hard cases (svara marks, verse numbers in parts or missing a danda, kampa digits, joiners, avagraha), and no label anywhere keeps a mark, digit or danda |
| `test_pipeline.py` | Pavamana's 610 verse units and their labels; any text's blocks as units, and a recording's parts in order; recordings name no one; the metrics ignore marks and find the nearest verse; a recording of several texts is chanted in order and scored against all of them; a results file names the code and libraries and never overwrites another; the aligner places units in order (skipped where PyTorch isn't installed, as in CI) |
| `test_follow.py` | `shravan/follow.py` gives what Listen's own code gives (`fixtures/listen.json`): the letters, lining up, matching and locating, and every text laid out as the site lays it out; one hearing follows near the mark, finds a lost place and waits through the dhrupad; Listen's mark while following, and whether a place found is right |
| `test_sound.py` | `normalise()` gives what the site's `normalize()` gives, and no other copy of it is left |
| `test_ctc.py` | The CTC model's letters: a label as it writes it, its alphabet with our letters added, encoding and decoding |
| `test_config.py` | The audio is outside the repository; `work/`, audio and model files are kept out of git; the held-out and validation hymns are real, don't overlap, leave enough to train on, and every hymn has exactly one split; training runs in full precision and fits a 16 GB Mac (checkpointing, the GPU memory cap) |

## Listen's answers

`fixtures/listen.json` is what the site's own code (Nitya Upasana, in the folder beside this one) answers for a few
cases and how it lays out every text of `texts/`. After a change to the site's `js/core/align.js` or
`js/core/follow.js`, or after `scripts/sync_texts.py`, make it again and commit it:

```sh
node tests/fixtures/listen.mjs > tests/fixtures/listen.json
```

## What they can't cover

How well a model hears. That is measured by testing it on the held-out recordings, on the Mac, and kept in
`results/`.
