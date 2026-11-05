"""Score a model on the clips of one split: letters wrong and verses found (shravan/metrics.py).

    python scripts/evaluate.py openai/whisper-base              # the untrained model, on the test clips
    python scripts/evaluate.py checkpoints/whisper-base --split validation

Writes one file to results/, with a few examples of what the model wrote beside what was chanted.
"""

import argparse
import time

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from shravan import config, data, env, metrics, results, texts, units


def transcribe(model_name: str, paths: list[str], language: str, batch: int = 8) -> list[str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    out: list[str] = []
    for i in range(0, len(paths), batch):
        audio = [sf.read(p, dtype="float32")[0] for p in paths[i : i + batch]]
        feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=225)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    args = ap.parse_args()

    cfg = config.load()
    env.cap_gpu_memory(cfg.training["gpu_memory"])
    clips = data.clips(cfg.work, args.split)
    every = units.verses(texts.load_all(cfg.texts)["vedic-chant-pavamana-suktam"])
    labels = [u.label for u in every]
    index = {u.id: i for i, u in enumerate(every)}

    started = time.time()
    heard = transcribe(args.model, [str(c.path) for c in clips], cfg.language)
    seconds = time.time() - started
    scores = {
        "split": args.split,
        "clips": len(clips),
        "minutes": round(sum(c.seconds for c in clips) / 60, 1),
        "letters_wrong": round(metrics.letters_wrong([c.label for c in clips], heard), 4),
        "verses_found": round(metrics.verses_found([index[c.id] for c in clips], heard, labels), 4),
        "seconds_to_transcribe": round(seconds, 1),
    }
    examples = [{"id": c.id, "chanted": c.label, "heard": h} for c, h in list(zip(clips, heard, strict=True))[:5]]
    path = results.save("evaluate", args.model, cfg, scores, {"examples": examples})
    print(scores)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
