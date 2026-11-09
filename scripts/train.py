"""Fine-tune Whisper on the training clips, keeping the checkpoint that does best on the validation clips.

    python scripts/train.py openai/whisper-base
    python scripts/train.py openai/whisper-small

Every setting comes from config.yaml (training:). The best model is saved to checkpoints/<model>/best (never in git),
and the run's settings and validation scores to results/. The test clips are not touched: score the saved model on
them with scripts/evaluate.py.
"""

import argparse
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from shravan import config, data, env, metrics, recordings, results
from shravan.augment import augment
from shravan.config import ROOT
from shravan.gpu import FreeMemory

LABEL_TOKENS = 128


# (torch is untyped to mypy, so subclassing its Dataset needs the ignore)
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
    processor: WhisperProcessor
    decoder_start: int
    train: bool
    rng: random.Random

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [sf.read(b["path"], dtype="float32")[0] for b in batch]
        if self.train:
            audio = [augment(a, self.rng) for a in audio]
        feats = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        # one label length for every batch (the longest label is about 105 tokens): on the Mac's GPU each new shape
        # keeps its own memory, and lengths changing batch by batch grew the run to 12 GB
        tok = self.processor.tokenizer(
            [b["label"] for b in batch],
            padding="max_length",
            max_length=LABEL_TOKENS,
            truncation=True,
            return_tensors="pt",
        )
        labels = tok.input_ids.masked_fill(tok.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start).all():  # the model adds the start token itself
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def per_model(setting: Any, model: str) -> Any:
    """A training setting is one value, or one per model name (config.yaml)."""
    return setting[model] if isinstance(setting, dict) else setting


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

    processor = WhisperProcessor.from_pretrained(args.model)
    # the labels must start with the same language and task tokens generation uses (<|sa|><|transcribe|>); passing
    # language= to from_pretrained does not set them in this version of transformers
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task="transcribe")
    # PyTorch's own attention, which on the Mac's GPU needs far less memory than writing out every attention matrix
    model = WhisperForConditionalGeneration.from_pretrained(args.model, attn_implementation="sdpa")
    model.generation_config.language = cfg.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = not t["checkpointing"]  # the decoder's cache can't be kept while recomputing

    train, valid = data.clips(cfg.work, "train"), data.clips(cfg.work, "validation")
    # a family recording's clips count family_repeat times (config.yaml); a published one's once
    family = {r.slug for r in recordings.load(cfg.recordings) if r.who != "published"}
    train = data.weighed(train, family, t["family_repeat"])
    start = model.config.decoder_start_token_id
    out = ROOT / "checkpoints" / (args.name or re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-"))

    def score(pred: Any) -> dict[str, float]:
        ids = np.where(pred.predictions != -100, pred.predictions, processor.tokenizer.pad_token_id)
        heard = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return {"letters_wrong": metrics.letters_wrong([c.label for c in valid], heard)}

    settings = Seq2SeqTrainingArguments(
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
        eval_steps=t["eval_every"],
        save_steps=t["eval_every"],
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
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
    trainer = Seq2SeqTrainer(
        model=model,
        args=settings,
        train_dataset=Clips(train),
        eval_dataset=Clips(valid),
        data_collator=Collate(processor, start, True, random.Random(t["seed"])),
        compute_metrics=score,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=per_model(t["patience"], args.model)), FreeMemory()],
    )
    # the validation clips are never augmented: they measure, they don't teach
    trainer.get_eval_dataloader = lambda ds=None: torch.utils.data.DataLoader(
        Clips(valid), batch_size=batch, collate_fn=Collate(processor, start, False, random.Random(0))
    )

    started = time.time()
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    best = out / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    history = [h for h in trainer.state.log_history if "eval_letters_wrong" in h]
    scores = {
        "train_clips": len(train),
        "family_clips": sum(c.recording in family for c in train),
        "validation_clips": len(valid),
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
