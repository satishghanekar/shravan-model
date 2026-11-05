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
| `test_config.py` | The audio is outside the repository; `work/`, audio and model files are kept out of git; the held-out hymns are real and leave enough to train on; training runs in full precision |

## What they can't cover

How well a model hears. That is measured by testing it on the held-out recordings, on the Mac, and kept in
`results/`.
