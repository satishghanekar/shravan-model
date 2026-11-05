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
    language: str
    heldout_hymns: tuple[int, ...]
    training: dict[str, Any]
    targets: dict[str, float]


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
        language=raw["language"],
        heldout_hymns=tuple(raw["heldout_hymns"]),
        training=dict(raw["training"]),
        targets=dict(raw["targets"]),
    )
