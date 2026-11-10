"""The settings in config.yaml, read once and typed, so no script hard-codes a path, model name or number."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    texts: Path
    audio: Path
    work: Path
    recordings: Path
    model: str
    heldout_hymns: tuple[int, ...]
    validation_hymns: tuple[int, ...]
    min_alignment_score: float
    min_alignment_score_family: float
    training: dict[str, Any]
    targets: dict[str, float]
    listen: dict[str, float]


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(path: Path = ROOT / "config.yaml") -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(
        texts=_path(raw["texts"]),
        audio=_path(raw["audio"]),
        work=_path(raw["work"]),
        recordings=_path(raw["recordings"]),
        model=raw["model"],
        heldout_hymns=tuple(raw["heldout_hymns"]),
        validation_hymns=tuple(raw["validation_hymns"]),
        min_alignment_score=float(raw["min_alignment_score"]),
        min_alignment_score_family=float(raw["min_alignment_score_family"]),
        training=dict(raw["training"]),
        targets=dict(raw["targets"]),
        listen=dict(raw["listen"]),
    )


def overrides(pairs: list[str]) -> dict[str, Any]:
    """key=value pairs as settings, the values read as config.yaml reads them (true, 4, 1.0e-5)."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"not key=value: {pair!r}")
        out[key] = yaml.safe_load(value)
    return out
