"""Play each tested recording as Listen hears it and score how often it finds the right place: the app's own measure,
and the one the release target is set on (README, "The targets").

    python scripts/follow.py checkpoints/ctc-pro-all/best
    python scripts/follow.py models/ctc-v4-q4

For every unit tested (a held-out recording's units, or a training recording's test hymns), 2, 3 and 5 seconds after
it starts, the model hears the last `window` seconds (config.yaml, listen:, as the site has it) and shravan/follow.py
finds the place exactly as Listen does. Two numbers each time:

- following: Listen followed the chanting until the unit before this one, and its mark moved on from there at its
  starting pace (or waited, through the dhrupad): the window is lined up with the text near the mark, and its letters
  looked for when that doesn't fit (the app's usual case);
- search: nothing followed yet, so the letters heard are looked for in every text, as Listen does when it is lost.

A place is right when it is in the unit being chanted; while an aarti's dhrupad is said again, when the mark waits or
the dhrupad is found. (Listen also waits to hear another text twice before it moves there; a place found in another
text here simply counts as wrong.) Only a CTC model (the site's): Whisper gives no scores to line up. Writes one file
to results/.
"""

import argparse
import json

import soundfile as sf

from shravan import config, env, follow, hear, recordings, results, texts
from shravan.split import split_of

AFTER = (2.0, 3.0, 5.0)  # seconds after a unit starts at which its place is checked


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    cfg = config.load()
    env.cap_gpu_memory(min(cfg.training["gpu_memory"].values()))
    if not hear.is_ctc(args.model):
        ap.error("Listen follows a CTC model's scores: give a CTC model (wav2vec2)")
    s = follow.Settings.of(cfg.listen)
    every = texts.load_all(cfg.texts)
    model = hear.Ctc(args.model)
    m = model.follow_model()

    scores: dict[str, object] = {"listen": cfg.listen}
    for rec in recordings.load(cfg.recordings):
        timings = cfg.work / "align" / f"{rec.slug}.json"
        if not timings.exists():
            continue
        rows = json.loads(timings.read_text(encoding="utf-8"))
        # the units tested: aligned well, and never trained on
        tested = [
            i
            for i, r in enumerate(rows)
            if r["score"] >= cfg.min_alignment_score and (rec.heldout or split_of(r["hymn"], cfg) == "test")
        ]
        if not tested:
            continue
        # every text, as Listen searches the whole library when it is lost: the recording's own first
        book = follow.book([every[t] for t in dict.fromkeys([*rec.texts, *every])])
        audio, rate = sf.read(cfg.work / "audio" / f"{rec.slug}.wav", dtype="float32")
        out: dict[str, object] = {"units": len(tested)}
        for after in AFTER:
            search = following = 0
            for i in tested:
                r = rows[i]
                # (never past the unit's own end: a short unit is checked at its end)
                t = r["start"] + min(after, r["end"] - r["start"])
                heard = model.scores(audio[int(max(t - s.window, 0) * rate) : int(t * rate)])
                search += follow.right(book, r["id"], follow.hear(heard, book.docs, None, m, s))
                mark = follow.mark_at(book, rows, i, t, s)
                following += follow.right(book, r["id"], follow.hear(heard, book.docs, mark, m, s))
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
