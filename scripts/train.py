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

import librosa
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

from shravan import config, data, env, metrics, results
from shravan.config import ROOT


# (torch is untyped to mypy, so subclassing its Dataset needs the ignore)
class Clips(torch.utils.data.Dataset[dict[str, Any]]):  # type: ignore[misc]
    def __init__(self, clips: list[data.Clip]) -> None:
        self.clips = clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int) -> dict[str, Any]:
        c = self.clips[i]
        return {"path": str(c.path), "label": c.label}


def augment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random speed (±10%, which also shifts the pitch a little), volume and quiet noise."""
    rate = rng.uniform(0.9, 1.1)
    audio = librosa.resample(audio, orig_sr=16000, target_sr=int(16000 / rate))
    audio = audio * rng.uniform(0.6, 1.2)
    audio = audio + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 0.004), audio.shape)
    return audio.astype(np.float32)


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
        tok = self.processor.tokenizer([b["label"] for b in batch], padding=True, return_tensors="pt")
        labels = tok.input_ids.masked_fill(tok.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start).all():  # the model adds the start token itself
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    cfg = config.load()
    t = cfg.training
    rate = t["learning_rate"][args.model] if isinstance(t["learning_rate"], dict) else t["learning_rate"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])

    processor = WhisperProcessor.from_pretrained(args.model)
    # the labels must start with the same language and task tokens generation uses (<|sa|><|transcribe|>); passing
    # language= to from_pretrained does not set them in this version of transformers
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = cfg.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    train, valid = data.clips(cfg.work, "train"), data.clips(cfg.work, "validation")
    start = model.config.decoder_start_token_id
    out = ROOT / "checkpoints" / re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")

    def score(pred: Any) -> dict[str, float]:
        ids = np.where(pred.predictions != -100, pred.predictions, processor.tokenizer.pad_token_id)
        heard = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return {"letters_wrong": metrics.letters_wrong([c.label for c in valid], heard)}

    settings = Seq2SeqTrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        learning_rate=rate,
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["patience"])],
    )
    # the validation clips are never augmented: they measure, they don't teach
    trainer.get_eval_dataloader = lambda ds=None: torch.utils.data.DataLoader(
        Clips(valid), batch_size=t["batch_size"], collate_fn=Collate(processor, start, False, random.Random(0))
    )

    started = time.time()
    trainer.train()
    best = out / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    history = [h for h in trainer.state.log_history if "eval_letters_wrong" in h]
    scores = {
        "train_clips": len(train),
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
