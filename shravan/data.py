"""The clips made by scripts/cut.py, read from every recording's manifest in work/clips/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    id: str
    path: Path
    label: str
    hymn: int
    split: str
    seconds: float
    recording: str


def clips(work: Path, split: str | None = None) -> list[Clip]:
    out = []
    for manifest in sorted((work / "clips").glob("*/manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if split is None or r["split"] == split:
                clip = Clip(
                    r["id"],
                    work / "clips" / r["file"],
                    r["label"],
                    r["hymn"],
                    r["split"],
                    r["seconds"],
                    manifest.parent.name,
                )
                out.append(clip)
    return out
