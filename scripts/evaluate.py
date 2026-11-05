"""Score a model on the clips of one split: letters wrong and verses found (shravan/metrics.py), for each recording.

    python scripts/evaluate.py openai/whisper-base              # the untrained model, on the test clips
    python scripts/evaluate.py checkpoints/whisper-base --split validation

Each recording is scored on its own (a family recording of a text never trained on is a different question from the
held-out hymns of a training recording); a clip's verse is looked for among every unit of its own text. Writes one
file to results/, with a few examples of what the model wrote beside what was chanted.
"""

import argparse
import json
import time

import soundfile as sf

from shravan import config, data, env, hear, metrics, results, texts, units


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    args = ap.parse_args()

    cfg = config.load()
    env.cap_gpu_memory(min(cfg.training["gpu_memory"].values()))  # testing needs far less than training
    clips = data.clips(cfg.work, args.split)
    every = texts.load_all(cfg.texts)

    started = time.time()
    heard = hear.transcribe(args.model, [sf.read(c.path, dtype="float32")[0] for c in clips], cfg.language)
    seconds = time.time() - started
    scores: dict[str, object] = {"split": args.split, "seconds_to_transcribe": round(seconds, 1)}
    examples = []
    for rec in sorted({c.recording for c in clips}):
        pairs = [(c, h) for c, h in zip(clips, heard, strict=True) if c.recording == rec]
        own = units.of(every[pairs[0][0].id.split("/")[0]])
        index = {u.id: i for i, u in enumerate(own)}
        scores[rec] = {
            "clips": len(pairs),
            "minutes": round(sum(c.seconds for c, _ in pairs) / 60, 1),
            "letters_wrong": round(metrics.letters_wrong([c.label for c, _ in pairs], [h for _, h in pairs]), 4),
            "verses_found": round(
                metrics.verses_found([index[c.id] for c, _ in pairs], [h for _, h in pairs], [u.label for u in own]), 4
            ),
            "out_of": len(own),
        }
        examples += [{"id": c.id, "chanted": c.label, "heard": h} for c, h in pairs[:5]]
    path = results.save("evaluate", args.model, cfg, scores, {"examples": examples})
    print(json.dumps(scores, indent=1))
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
