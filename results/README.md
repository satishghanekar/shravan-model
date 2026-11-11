# How the models got here

The results of every run are the files in this folder; this is the story they tell, newest last. The model on the site
today, and how to rebuild it, is in the main README ("The current model").

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

Trained on one published recording of one text, the model already follows a family voice through another.

Then trained again with SG's recordings at home as well (21 minutes, each clip counted three times;
`checkpoints/base-v2-family`), on tests it never trained on:

| Whisper base | Pavamana 61–63 | SG's Purusha Suktam | SJ's Lalita |
|---|---|---|---|
| Published Pavamana only: letters wrong | 10.0% | 29.2% | 45.1% |
| With SG's recordings: letters wrong | 10.3% | **24.6%** | **40.9%** |
| With SG's recordings: right place within 3 s (following) | 98.7% | 100% | 96.6% |

Shrunk for the phone (`scripts/export.py`, ONNX with int8 weights: 76 MB instead of 290), v2 hears as well: 26.1%,
40.8% and 10.8% letters wrong, the right place within 3 s following 100%, 96.6% and 98.7%. One 6-second window takes
about 1.1 s on one of the M4's CPU cores. Next: how fast it runs on an iPhone, in the browser.

On an iPhone (4 cores, Safari, 2026-11-06) Whisper took about 1.7 s a window, too slow to follow along by itself. So a
CTC model (`scripts/train_ctc.py`): Vakyansh's Sanskrit wav2vec2 base, which writes a letter for every 20 ms and hears
only the sound it is given, fine-tuned on the same clips (`checkpoints/ctc-sam60`), on the same tests:

| | Pavamana 61–63 | SG's Purusha Suktam | SJ's Lalita |
|---|---|---|---|
| Untrained CTC: letters wrong | 40.4% | 23.8% | 35.9% |
| Trained CTC: letters wrong | 11.3% | **15.1%** | 37.7% |
| Trained CTC: right place within 3 s (following) | 100% | 100% | 96.6% |
| Shrunk (int8, 122 MB): letters wrong | 11.5% | 15.1% | 38.1% |
| Shrunk, positional convolution too (108 MB, the site's `chant-v3`): letters wrong | 11.9% | 15.4% | 38.5% |
| … right place within 3 s (following) | 100% | 100% | 96.6% |

It hears as well as Whisper (better on SG's voice), and much faster: on the M4's CPU with 4 threads, 58 ms for 2 s of
sound against Whisper's 590 ms a window (`scripts/speed.py`); about 0.4 s on an iPhone, by the encoder's ratio. Its
pretrained model's "nothing said" token is `<s>`, not the `<pad>` its config names: trained against the wrong one it
wrote nothing within a few dozen steps, so `train_ctc.py` finds it from what the model writes.

Its positional convolution (19 MB) kept its weight as two parts multiplied as it runs, which the quantizer can't
shrink; `export.py` works it out once first (2026-11-07), 108 MB in all. Shrinking the convolutions that hear the raw
sound too (95 MB) cost too much: 19.3%, 23.6% and 40.9% letters wrong (`results/*-every-conv-shrunk.json`).
`scripts/to_site.py` packs it for the site (`models/site/chant-v3`): its pieces, each output's letter, and the scores
of a check sound, which the site's own run of the model must match.

Two more ways to shrink it, tried on the same tests (2026-11-07) and not taken:

| | Pavamana 61–63 | SG's Purusha Suktam | SJ's Lalita | Size | 4 s in the browser / 5 s here |
|---|---|---|---|---|---|
| The site's `chant-v3` (12 layers, int8): letters wrong | 11.9% | 15.4% | 38.5% | 108 MB | 373 ms / 140 ms |
| Weights in 4 bits (`MatMulNBits`, the sound layers as before) | 11.9% | 15.7% | 38.0% | 76 MB | 421 ms / 208 ms |
| 8 layers of 12, trained again (`train_ctc.py --layers 8`, `checkpoints/ctc-sam60-8`) | 12.5% | 19.9% | 43.9% | 93 MB | – / 106 ms |

Both found the right place within 3 s as often (100%, 100%, 96.6%). Four bits hear as well but run slower in the
browser's WebAssembly (Chromium on the M4, 4 threads, the site's own code). Eight layers are a quarter quicker but
hear the family's voices clearly worse, and only 15 MB smaller: the convolutions that hear the raw sound stay full
size (17 MB).

### More recordings (2026-11-08)

Added: two more published recordings by the Challakere Brothers (the Pavamana Suktam; the Atharvashirsha, Nyasam,
Namakam, Chamakam and Durga Suktam) and SG's second takes at home (the whole Namakam and Chamakam, the Sunday aartis,
the Durga Suktam, Atharvashirsha, Laghunyasa and Shri Sukta). Training went from about 60 to 229 minutes of clips.
Newly held out: a published Lalita Sahasranama in one voice (28 minutes, beside SJ's group recording) and SG's second
Purusha Suktam. A third published recording's Purusha Suktam matched the site's text too poorly to use (median
alignment 0.21: another version of the hymn) and was left out. Every model trained on the same clips, was chosen on
the same validation clips and tested on the same held-out audio; letters wrong:

| | Pavamana 61–63, the first published recording | Pavamana 61–63, Challakere | SG's Purusha, 2nd take | SG's Purusha, 1st take | Lalita, one voice | SJ's Lalita, a group |
|---|---|---|---|---|---|---|
| The site's `chant-v3` | 11.9% | 11.4% | 19.5% | 15.4% | 11.3% | 38.5% |
| CTC, every recording (`checkpoints/ctc-pro-all`) | 13.1% | 6.5% | 17.9% | **14.8%** | 10.7% | **36.4%** |
| … shrunk (int8, 103 MB: `models/ctc-v3-int8`) | 13.4% | 6.3% | **17.6%** | 15.3% | **10.3%** | 36.7% |
| CTC, published recordings only (`--only published`) | **10.2%** | **5.2%** | 24.0% | 18.8% | 11.9% | 39.1% |
| CTC, 8 layers of the first (`--layers 8`), shrunk (76 MB) | 14.5% | 8.4% | 20.5% | 17.9% | 14.5% | 42.6% |
| Whisper base, every recording | 10.6% | 5.4% | 28.1% | 23.9% | 16.9% | 39.3% |

Verses found and the right place within 3 s are as before for all of them (100%, but 96.6% for SJ's; searching the
whole text, `ctc-v3-int8` and the 8 layers miss one place in 23 of SG's 2nd take).

- SG's recordings are what teach SG's voice: trained on the published ones alone, SG's Purusha Suktam is 24% wrong
  against 18%. More published chanting teaches Pavamana (5–6% on a voice never heard), and a little of everything else.
- Every recording, 12 layers is the best for the site: better than `chant-v3` on five tests of six (the 1st take
  about the same), worse on the first published recording, which `chant-v3` learned from almost alone. Shrinking it
  cost nothing.
- Eight layers are worse than `chant-v3` on every family test; Whisper is worse on SG's voice and twice as slow.
- Clips of many lengths filled the GPU memory cap (`gpu_memory`): PyTorch keeps what each new shape of batch used.
  `shravan/gpu.py` hands it back every 50 steps; the 8-layer run needed a cap of 0.85 as well.

What Listen costs, measured in the browser (Chromium on the M4, the site's own code and security policy, 4 threads,
the held-out one-voice Lalita played in as the microphone for 1–5 minutes):

| | 4 s heard (the model alone) | Cores busy while listening (the page idle: 0.3) | Memory |
|---|---|---|---|
| `chant-v3`, and `ctc-v3-int8` | 380 ms | 2.9 | ONNX Runtime's 233 MB; the tab grows by about 560 MB, kept after Stop |
| 8 layers | 323 ms | 2.6 | about the same |

Listen hears 4 s every half second, or as soon as the model is free, so the model works about three quarters of the
time, on most of 4 cores, for as long as Listen is on.

### On the phone's GPU (2026-11-09)

The same model (`checkpoints/ctc-pro-all`) for WebGPU, timed on an iPhone (Safari 27, 4 cores) by a test page (the
site's `claude/gpu-lab` branch preview): ONNX Runtime Web 1.30.0, the site's 8-bit model on the CPU as Listen ran it,
and its WebGPU build. Letters wrong on the held-out recordings (`scripts/evaluate.py`):

| chant-v4 | A 4 s hearing on the iPhone | Size | Pavamana, 1st published | Pavamana, Challakere | SG's Purusha, 2nd | SG's Purusha, 1st | Lalita, one voice | SJ's Lalita |
|---|---|---|---|---|---|---|---|---|
| 8-bit, CPU (`ctc-v3-int8`, Listen until 1.48) | 712 ms | 103 MB | 13.4% | 6.3% | 17.6% | 15.3% | 10.3% | 36.7% |
| 8-bit, GPU | 825 ms | 103 MB | | | | | | |
| **4-bit, GPU** (`export.py --q4`, `ctc-v4-q4`) | **233 ms** | 90 MB | 13.9% | 6.6% | 18.0% | 15.1% | 10.5% | 36.0% |
| 16-bit, GPU (`ctc-v4-fp16`) | 142 ms | 189 MB | 13.1% | 6.5% | 17.9% | 14.7% | 10.7% | 36.5% |

All three pick the same letters as the model's Python side on the check sound, and find the same verses. 4-bit is
the site's GPU model (`chant-v4-q4`): three times quicker than the CPU, a smaller download, and inside the half second
between hearings with room to spare; 16-bit is quicker still but twice the size. The GPU's 8-bit is no quicker than the
CPU (ONNX Runtime's WebGPU build is made for 4-bit and 16-bit weights; its 8-bit steps are not). On the M4 in Chrome, Listen on
the GPU took 86 ms a hearing and 0.3 cores, against 390 ms and about 3 on the CPU.

### chant-v4 by Listen's own method (2026-11-09)

`scripts/follow.py`, 3 seconds into each unit (`models/ctc-v3-int8` is chant-v4 on the CPU, `models/ctc-v4-q4` on the GPU):

| Recording (units tested) | Following, CPU | Following, GPU | Search, CPU | Search, GPU |
| --- | --- | --- | --- | --- |
| Pavamana, 1st published (79) | 100% | 100% | 96.2% | 94.9% |
| Pavamana, Challakere (90) | 100% | 100% | 95.6% | 94.4% |
| SG's Purusha, 2nd (23) | 100% | 100% | 82.6% | 87.0% |
| SG's Purusha, 1st (25) | 100% | 100% | 92.0% | 92.0% |
| Lalita, one voice (184) | 100% | 100% | 99.5% | 99.5% |
| SJ's Lalita, a group (58) | 98.3% | 98.3% | 48.3% | 48.3% |

Following, the number the release target is set on, passes on every recording. Search is what happens after a jump to
another text, when Listen has lost its place, and it is weakest on a group and on family voices. None of these is an
aarti, an ashtak or Marathi: both aarti recordings and the ashtak are trained on, so no score here says how Listen does
on them.

### Family voices choosing the checkpoint, batches of one length, 6000 steps (2026-11-10)

`ctc-family-sorted`, against chant-v4, on the same held-out recordings. Trained 6000 steps (4 hours; the best at 5600),
the last tenth of each family recording kept out to help choose the checkpoint, batches of clips of about one length.
3 seconds into each unit, on the CPU, chant-v4 first:

| Recording (units tested) | Letters wrong | Following | Search |
| --- | --- | --- | --- |
| Pavamana, 1st published (79) | 13.4% → 10.0% | 100% → 100% | 96.2% → 98.7% |
| Pavamana, Challakere (90) | 6.3% → 5.4% | 100% → 100% | 95.6% → 94.4% |
| SG's Purusha, 2nd (23) | 17.6% → 17.3% | 100% → 95.7% | 82.6% → 82.6% |
| SG's Purusha, 1st (25) | 15.3% → 14.8% | 100% → 100% | 92.0% → 88.0% |
| Lalita, one voice (184) | 10.3% → 10.9% | 100% → 100% | 99.5% → 98.9% |
| SJ's Lalita, a group (58) | 36.7% → 36.9% | 98.3% → 96.6% | 48.3% → 48.3% |

On the GPU (4-bit) the letters moved the same way. Search on a group fell from 48.3% to 41.4%, and the other search
and following figures stayed within a few units.

Published voices improved; family voices barely did, and following is one unit worse on two recordings (one unit is
4.3% of SG's 2nd take, 1.7% of the group). Not released: nothing here helps Listen follow a family voice, and the letters
target is still missed on them. Doubling the steps bought little: validation was 11.7% at step 3000 and 11.0% at 5600.

The same run with unsorted batches, meant to show what sorting alone does, ran out of the Mac's GPU memory at step 20 and
was not repeated. So these figures cannot say which of the three changes did what. The sorted run itself went all 6000
steps without running out of memory, which unsorted runs have often not done.
