"""Fine-tune a CTC model (wav2vec2: a letter for every 20 ms) on the training clips, keeping the checkpoint that does
best on the validation clips: the same clips, splits and family weighting as scripts/train.py for Whisper.

    python scripts/train_ctc.py Harveenchadha/vakyansh-wav2vec2-sanskrit-sam-60

The pretrained model already writes Devanagari letter by letter (Sanskrit); letters our labels use that it lacks are
added to its output (shravan/ctc.py), each starting from the average of the others. Its first layers, which turn sound
into features, stay as they are (the usual way to fine-tune wav2vec2 on little data). Every setting comes from
config.yaml (training:); the best model goes to checkpoints/<name>/best, the run's settings and scores to results/.
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

# CTC's loss isn't on the Mac's GPU in every PyTorch: that one step falls back to the CPU instead of failing
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from shravan import config, ctc, data, env, metrics, recordings, results  # noqa: E402
from shravan.augment import augment  # noqa: E402
from shravan.config import ROOT  # noqa: E402

# every batch's sound is padded up to a whole number of these (16 kHz samples, 4 s), so the Mac's GPU sees few shapes:
# each new shape keeps memory of its own (scripts/train.py, the labels)
BUCKET = 4 * 16000
LABEL_LETTERS = 400  # the longest label is about 300 letters


class Clips(torch.utils.data.Dataset[dict[str, Any]]):  # type: ignore[misc]
    def __init__(self, clips: list[data.Clip]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict[str, Any]:
        c = self.clips[i]
        return {"path": str(c.path), "label": c.label}


@dataclass
class Collate:
    alpha: dict[str, int]
    train: bool
    rng: random.Random

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [sf.read(b["path"], dtype="float32")[0] for b in batch]
        if self.train:
            audio = [augment(a, self.rng) for a in audio]
        # each clip brought to mean 0 and spread 1, as the model was trained (do_normalize), then padded with silence
        audio = [(a - a.mean()) / np.sqrt(a.var() + 1e-7) for a in audio]
        size = -(-max(len(a) for a in audio) // BUCKET) * BUCKET
        values = torch.zeros(len(audio), size)
        for i, a in enumerate(audio):
            values[i, : len(a)] = torch.from_numpy(a)
        labels = torch.full((len(batch), LABEL_LETTERS), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = ctc.encode(b["label"], self.alpha)[:LABEL_LETTERS]
            labels[i, : len(ids)] = torch.tensor(ids)
        return {"input_values": values, "labels": labels}


def per_model(setting: Any, model: str) -> Any:
    return setting[model] if isinstance(setting, dict) else setting


def widen(model: Wav2Vec2ForCTC, size: int) -> None:
    """The output layer given rows for the added letters, each the average of the pretrained ones."""
    old = model.lm_head
    if size == old.out_features:
        return
    new = torch.nn.Linear(old.in_features, size)
    with torch.no_grad():
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        new.weight[old.out_features :] = old.weight.mean(0)
        new.bias[old.out_features :] = old.bias.mean()
    model.lm_head = new
    model.config.vocab_size = size


def blank_of(model: Wav2Vec2ForCTC, clips: list[data.Clip]) -> int:
    """The model's "nothing said" token: the one it writes most on real chanting. Its config can't be trusted for it:
    Vakyansh's models, converted from fairseq, name <pad> but write <s> (id 0), and training against the wrong one
    taught the model that every pause was a mistake, so within a few dozen steps it wrote nothing at all
    (2026-11-06)."""
    counts: dict[int, int] = {}
    model.eval()
    for c in clips:
        a = sf.read(c.path, dtype="float32")[0]
        x = torch.from_numpy((a - a.mean()) / np.sqrt(a.var() + 1e-7))[None]
        with torch.inference_mode():
            for i in model(input_values=x).logits.argmax(-1)[0].tolist():
                counts[i] = counts.get(i, 0) + 1
    model.train()
    return max(counts, key=lambda i: counts[i])


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--resume", action="store_true", help="carry on from the last checkpoint of a stopped run")
    ap.add_argument("--name", help="its folder in checkpoints/ (default: the model name)")
    args = ap.parse_args()

    cfg = config.load()
    env.cap_gpu_memory(per_model(cfg.training["gpu_memory"], args.model))
    t = cfg.training
    rate, batch, accumulate = (per_model(t[k], args.model) for k in ("learning_rate", "batch_size", "accumulate"))
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])

    train, valid = data.clips(cfg.work, "train"), data.clips(cfg.work, "validation")
    family = {r.slug for r in recordings.load(cfg.recordings) if r.who != "published"}
    train += [c for c in train if c.recording in family] * (t["family_repeat"] - 1)

    pretrained = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
    alpha = ctc.alphabet(pretrained.get_vocab(), [c.label for c in train + valid])
    model = Wav2Vec2ForCTC.from_pretrained(args.model, ctc_loss_reduction="mean", ctc_zero_infinity=True)
    blank = blank_of(model, valid[:4])
    widen(model, len(alpha))
    model.config.pad_token_id = blank
    model.freeze_feature_encoder()
    out = ROOT / "checkpoints" / (args.name or re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-"))

    def choices(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # only each step's likeliest letter is kept from an evaluation, not every letter's score
        return logits.argmax(-1)

    def score(pred: Any) -> dict[str, float]:
        heard = [ctc.decode(row.tolist(), alpha, blank) for row in pred.predictions]
        return {"letters_wrong": metrics.letters_wrong([c.label for c in valid], heard)}

    settings = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accumulate,
        gradient_checkpointing=t["checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=rate,
        warmup_steps=per_model(t["warmup_steps"], args.model),
        max_steps=per_model(t["max_steps"], args.model),
        eval_strategy="steps",
        eval_steps=per_model(t["eval_every"], args.model),
        save_steps=per_model(t["eval_every"], args.model),
        save_total_limit=2,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="letters_wrong",
        greater_is_better=False,
        seed=t["seed"],
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=settings,
        train_dataset=Clips(train),
        eval_dataset=Clips(valid),
        data_collator=Collate(alpha, True, random.Random(t["seed"])),
        compute_metrics=score,
        preprocess_logits_for_metrics=choices,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=per_model(t["patience"], args.model))],
    )
    # the validation clips are never augmented, and come in their own order
    trainer.get_eval_dataloader = lambda ds=None: torch.utils.data.DataLoader(
        Clips(valid), batch_size=batch, collate_fn=Collate(alpha, False, random.Random(0))
    )

    started = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    best = out / "best"
    trainer.save_model(str(best))
    # the processor that goes with it: the widened alphabet, and the pretrained sound settings
    (best / "vocab.json").write_text(json.dumps(alpha, ensure_ascii=False), encoding="utf-8")
    # (its pad token is the blank, as shravan/hear.py reads it)
    letters = {i: ch for ch, i in alpha.items()}
    tokenizer = Wav2Vec2CTCTokenizer(str(best / "vocab.json"), word_delimiter_token=ctc.WORD, pad_token=letters[blank])
    Wav2Vec2Processor(Wav2Vec2FeatureExtractor.from_pretrained(args.model), tokenizer).save_pretrained(str(best))
    history = [h for h in trainer.state.log_history if "eval_letters_wrong" in h]
    scores = {
        "train_clips": len(train),
        "family_clips": sum(c.recording in family for c in train),
        "validation_clips": len(valid),
        "letters_added": len(alpha) - len(pretrained.get_vocab()),
        "blank": letters[blank],
        "steps": trainer.state.global_step,
        "best_step": int(str(trainer.state.best_model_checkpoint).rsplit("-", 1)[-1]),
        "best_validation_letters_wrong": round(float(trainer.state.best_metric or 0), 4),
        "validation_history": [(h["step"], round(h["eval_letters_wrong"], 4)) for h in history],
        "minutes": round((time.time() - started) / 60, 1),
        "learning_rate": rate,
    }
    path = results.save("train", args.model, cfg, scores, {"saved_to": str(best.relative_to(ROOT))})
    print(scores)
    print(f"best model saved to {best}; results in {path}")


if __name__ == "__main__":
    main()
