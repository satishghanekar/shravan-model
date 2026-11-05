"""Play a recording as the app hears it and score how quickly it finds the right place: the app's own measure.

    python scripts/follow.py checkpoints/openai-whisper-base/best

For every unit tested (a held-out recording's units, or a training recording's test hymns), at 2, 3 and 5 seconds
after the unit starts, the model hears the last 6 seconds of audio (so the end of the unit before, as a phone
listening all along would) and shravan/follow.py says where the chanting is. Two numbers each time:

- search: the whole text searched, as when the app has lost its place;
- following: only the units around the one before (the app's usual case); a jump back in the text (the Dhyanam said
  again at the end) counts as a miss, which in the app falls back to a search.

The target (README): the right place within 3 seconds more than 95% of the time. Writes one file to results/.
"""

import argparse
import json

import soundfile as sf

from shravan import config, env, follow, hear, recordings, results, texts, units
from shravan.split import split_of

WINDOW = 6.0  # seconds of audio the app keeps listening to
AFTER = (2.0, 3.0, 5.0)  # seconds after a unit starts at which its place is checked


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    cfg = config.load()
    env.cap_gpu_memory(min(cfg.training["gpu_memory"].values()))
    every = texts.load_all(cfg.texts)

    scores: dict[str, object] = {"window_seconds": WINDOW}
    for rec in recordings.load(cfg.recordings):
        timings = cfg.work / "align" / f"{rec.slug}.json"
        if not timings.exists():
            continue
        rows = json.loads(timings.read_text(encoding="utf-8"))
        own = units.of(every[rec.texts[0]])
        index = {u.id: i for i, u in enumerate(own)}
        labels = [u.label for u in own]
        audio, rate = sf.read(cfg.work / "audio" / f"{rec.slug}.wav", dtype="float32")
        # the units tested: aligned well, and never trained on
        tested = [
            (i, r)
            for i, r in enumerate(rows)
            if r["score"] >= cfg.min_alignment_score and (rec.heldout or split_of(r["hymn"], cfg) == "test")
        ]
        if not tested:
            continue
        out: dict[str, object] = {"units": len(tested)}
        for after in AFTER:
            pieces = []
            for _, r in tested:
                # (never past the unit's own end: a short piece is checked at its end)
                t = r["start"] + min(after, r["end"] - r["start"])
                pieces.append(audio[int(max(t - WINDOW, 0) * rate) : int(t * rate)])
            heard = hear.transcribe(args.model, pieces, cfg.language)
            search = following = 0
            for (i, r), h in zip(tested, heard, strict=True):
                truth = index[r["id"]]
                search += follow.where(h, labels) == truth
                # following: the place before is known (the unit before this one in the recording)
                last = index[rows[i - 1]["id"]] if i else truth
                near = list(follow.nearby(last, len(own)))
                found = follow.where(h, [labels[j] for j in near])
                following += found >= 0 and near[found] == truth
            out[f"after_{after:g}s"] = {
                "search": round(search / len(tested), 4),
                "following": round(following / len(tested), 4),
            }
        scores[rec.slug] = out
        print(rec.slug, json.dumps(out))
    path = results.save("follow", args.model, cfg, scores)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
