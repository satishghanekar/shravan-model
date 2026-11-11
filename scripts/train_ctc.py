"""Fine-tune the model (wav2vec2: a letter for every 20 ms) on the training clips, keeping the checkpoint that does best
on the validation clips.

    python scripts/train_ctc.py
    python scripts/train_ctc.py checkpoints/ctc-sam60/best --layers 8 --name ctc-sam60-8

The first argument is where to start from (default: `model` in config.yaml). --layers keeps only the first layers of
the transformer (12 in a base model), and trains again from there: a smaller and quicker model for the phone, each
layer about 7 MB shrunk (2026-11-07). --only published (or family) trains on that kind of recording alone, to compare
what each teaches; the validation clips stay the same, so the runs choose alike. --set changes one training: setting
for a comparison (--set sort_by_length=false), recorded in the results file like the others.

The best checkpoint is the one with the fewest letters wrong on the validation clips, the published hymns and the last
part of each family recording counted alike (shravan/data.py, family_validation).

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
from transformers.trainer_pt_utils import LengthGroupedSampler  # noqa: E402

from shravan import config, ctc, data, env, metrics, recordings, results  # noqa: E402
from shravan.augment import augment  # noqa: E402
from shravan.config import ROOT  # noqa: E402
from shravan.gpu import FreeMemory  # noqa: E402
from shravan.sound import normalise  # noqa: E402

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
    # every batch's sound is padded up to a whole number of these (samples), so the Mac's GPU sees few shapes: each new
    # shape keeps memory of its own (the labels too, always LABEL_LETTERS long)
    bucket: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [sf.read(b["path"], dtype="float32")[0] for b in batch]
        if self.train:
            audio = [augment(a, self.rng) for a in audio]
        # each clip brought to mean 0 and spread 1, as the model was trained (do_normalize), then padded with silence
        audio = [normalise(a) for a in audio]
        size = -(-max(len(a) for a in audio) // self.bucket) * self.bucket
        values = torch.zeros(len(audio), size)
        for i, a in enumerate(audio):
            values[i, : len(a)] = torch.from_numpy(a)
        labels = torch.full((len(batch), LABEL_LETTERS), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = ctc.encode(b["label"], self.alpha)[:LABEL_LETTERS]
            labels[i, : len(ids)] = torch.tensor(ids)
        return {"input_values": values, "labels": labels}


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


def find_blank(model: Wav2Vec2ForCTC, clips: list[data.Clip]) -> int:
    """The model's "nothing said" token, from what it writes on a few clips (shravan/ctc.py, blank_of)."""
    chosen = []
    model.eval()
    for c in clips:
        a = sf.read(c.path, dtype="float32")[0]
        x = torch.from_numpy(normalise(a))[None]
        with torch.inference_mode():
            chosen.append(model(input_values=x).logits.argmax(-1)[0].tolist())
    model.train()
    return ctc.blank_of(chosen)


class ByLength(Trainer):  # type: ignore[misc]
    """Batches of clips of about one length, still shuffled from pass to pass (group_by_length needs a clip's sound in
    hand to know its length; ours are read a batch at a time)."""

    lengths: list[int]

    def _get_train_sampler(self, train_dataset: Any = None) -> Any:
        return LengthGroupedSampler(self.args.train_batch_size, lengths=self.lengths)


def main() -> None:
    env.use_certifi()
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="what to start from (default: model in config.yaml)")
    ap.add_argument("--resume", action="store_true", help="carry on from the last checkpoint of a stopped run")
    ap.add_argument("--name", help="its folder in checkpoints/ (default: the model name)")
    ap.add_argument("--layers", type=int, help="keep only this many of the transformer's first layers")
    ap.add_argument("--only", choices=["published", "family"], help="train on only this kind of recording")
    ap.add_argument("--set", action="append", default=[], help="change one training: setting, key=value")
    args = ap.parse_args()

    cfg = config.load()
    args.model = args.model or cfg.model
    t = cfg.training
    t.update(config.overrides(args.set))  # (results.save records these with the rest)
    env.cap_gpu_memory(t["gpu_memory"])
    rate, batch, accumulate = t["learning_rate"], t["batch_size"], t["accumulate"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])

    train, valid = data.clips(cfg.work, "train"), data.clips(cfg.work, "validation")
    family = {r.slug for r in recordings.load(cfg.recordings) if r.who != "published"}
    train, family_valid = data.family_validation(train, family, t["family_validation"])
    published = len(valid)
    valid = valid + family_valid
    train = data.weighed(train, family, t["family_repeat"], args.only)

    pretrained = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
    alpha = ctc.alphabet(pretrained.get_vocab(), [c.label for c in train + valid])
    model = Wav2Vec2ForCTC.from_pretrained(args.model, ctc_loss_reduction="mean", ctc_zero_infinity=True)
    if args.layers:
        model.wav2vec2.encoder.layers = model.wav2vec2.encoder.layers[: args.layers]
        model.config.num_hidden_layers = args.layers
    blank = find_blank(model, valid[:4])
    widen(model, len(alpha))
    model.config.pad_token_id = blank
    model.freeze_feature_encoder()
    out = ROOT / "checkpoints" / (args.name or re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-"))

    def choices(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # only each step's likeliest letter is kept from an evaluation, not every letter's score
        return logits.argmax(-1)

    def score(pred: Any) -> dict[str, float]:
        heard = [ctc.decode(row.tolist(), alpha, blank) for row in pred.predictions]
        said = [c.label for c in valid]
        one = metrics.letters_wrong(said[:published], heard[:published])
        other = metrics.letters_wrong(said[published:], heard[published:]) if family_valid else None
        best = metrics.validation_score(one, other)
        return {"letters_wrong": best, "published": one} | ({"family": other} if other is not None else {})

    settings = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accumulate,
        gradient_checkpointing=t["checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=rate,
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_every"],
        save_steps=t["eval_every"],
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
    bucket = int(t["pad_to_seconds"] * 16000)
    trainer = (ByLength if t["sort_by_length"] else Trainer)(
        model=model,
        args=settings,
        train_dataset=Clips(train),
        eval_dataset=Clips(valid),
        data_collator=Collate(alpha, True, random.Random(t["seed"]), bucket),
        compute_metrics=score,
        preprocess_logits_for_metrics=choices,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["patience"]), FreeMemory()],
    )
    # the validation clips are never augmented, and come in their own order
    trainer.get_eval_dataloader = lambda ds=None: torch.utils.data.DataLoader(
        Clips(valid), batch_size=batch, collate_fn=Collate(alpha, False, random.Random(0), bucket)
    )

    trainer.lengths = [round(c.seconds * 16000) for c in train]
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
        "family_validation_clips": len(family_valid),
        "letters_added": len(alpha) - len(pretrained.get_vocab()),
        "blank": letters[blank],
        "steps": trainer.state.global_step,
        "best_step": int(str(trainer.state.best_model_checkpoint).rsplit("-", 1)[-1]),
        "best_validation_letters_wrong": round(float(trainer.state.best_metric or 0), 4),
        "validation_history": [(h["step"], round(h["eval_letters_wrong"], 4)) for h in history],
        "validation_at_best": [
            {k[5:]: round(v, 4) for k, v in h.items() if k in ("eval_published", "eval_family")}
            for h in history
            if h["step"] == int(str(trainer.state.best_model_checkpoint).rsplit("-", 1)[-1])
        ],
        "minutes": round((time.time() - started) / 60, 1),
        "learning_rate": rate,
        "layers": model.config.num_hidden_layers,
        "trained_on": args.only or "all",
        "recordings_trained_on": sorted({c.recording for c in train}),
    }
    path = results.save("train", args.model, cfg, scores, {"saved_to": str(best.relative_to(ROOT))})
    print(scores)
    print(f"best model saved to {best}; results in {path}")


if __name__ == "__main__":
    main()
