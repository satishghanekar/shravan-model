# How Shravan fits together

Shravan makes one thing: a small speech model that turns family chanting into plain Devanagari, accurately enough that
the Nitya Upasana app can tell which line is being chanted. The app does the following along; the model only has to
hear the words. Everything here exists to train that model fairly and to prove how good it is.

## The pipeline

```
 site texts ──sync──▶ texts/ ──labels──▶ plain Devanagari per line
                                              │
 recordings (iCloud) ──align──▶ line timings ─┴─cut──▶ clips + labels (work/)
                                                            │
                                    ┌──── train split ──────┴────── held-out split ────┐
                                    ▼                                                   ▼
                          fine-tune Whisper on the M4                         test: letters wrong,
                                    │                                         lines found in 3 s
                                    └──────────▶ shrink (ONNX, int8) ──▶ test again ──▶ site
```

1. **Sync** (`scripts/sync_texts.py`): the texts are copied from the site, with the commit they came from.
2. **Labels** (`shravan/labels.py`): each line's plain Devanagari, the letters that are said and nothing else.
3. **Align** (to come): for each recording, where each line starts and ends. The text is known, so this is forced
   alignment (torchaudio's MMS aligner on romanized text), not recognition.
4. **Cut** (to come): line-sized clips of at most 30 seconds (Whisper's limit), each with its label, into `work/`.
5. **Split**: held-out hymns and held-out recordings (`config.yaml`, `recordings.csv`) never reach training.
6. **Test** (to come): the untrained model first, then each trained one, on the held-out clips only.
7. **Train** (to come): full fine-tuning of Whisper base (the phone model) and small (the accuracy ceiling), fp32 on
   the M4's GPU (`mps`), keeping the best checkpoint by held-out score.
8. **Shrink and release** (to come): ONNX with int8 weights (`optimum-onnx`), tested again, then copied into the site
   as `models/chant-vN/` by the site's own rules.

## Why these choices

- **Whisper.** Free (MIT licence), already knows a little Marathi and Sanskrit, and runs in a web page with
  Transformers.js. Base (74M parameters) fits a phone at about 80 MB once shrunk; small (244M) is the most the 16 GB
  Mac can fine-tune and shows how good base could get. Medium does not fit training in 16 GB.
- **Plain Devanagari labels.** The app compares letters, so svara marks, verse numbers and the like would only make
  the model's job harder without helping it follow along.
- **Held-out hymns.** With one long recording of one text, the danger is a model that memorises those takes. Testing
  on hymns it never trained on shows whether it hears chanting or replays it.
- **Two numbers.** Letters wrong (character error rate) says how well it hears; lines found within 3 seconds says
  whether the app can follow. The second decides a release.

## Where the data lives

| Data | Place | In git |
|---|---|---|
| Texts | `texts/` | yes: public already, on the site |
| Recording list | `recordings.csv` | yes: file names, dates, initials only |
| Recordings | iCloud Drive `development/shravan-training/` | never |
| Line timings, clips | `work/` | never: made from the recordings |
| Checkpoints, models | `checkpoints/`, `models/` | never: large, and rebuilt from the above |
| Results | `results/` | yes: the record of every run |

## What not to do

- Don't train on anything held out, even to "check quickly": the scores would stop meaning anything.
- Don't edit `texts/`; correct the site and sync.
- Don't hard-code paths, model names or numbers in a script; they belong in `config.yaml`.
- Don't compare scores across label-rule changes without retesting the untrained model.
