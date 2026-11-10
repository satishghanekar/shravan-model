"""The clips made by scripts/cut.py, read from every recording's manifest in work/clips/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def weighed(train: list[Clip], family: set[str], repeat: int, only: str | None = None) -> list[Clip]:
    """The training clips as a pass sees them: each family clip `repeat` times, so a few minutes of the voices the app
    must follow aren't drowned out by hours of published chanting. With `only` ("published" or "family"), the other
    kind is left out: to see what each kind of recording teaches on its own (2026-11-08)."""
    if only == "published":
        train = [c for c in train if c.recording not in family]
    elif only == "family":
        train = [c for c in train if c.recording in family]
    elif only is not None:
        raise ValueError(f"only 'published' or 'family', not {only!r}")
    return train + [c for c in train if c.recording in family] * (repeat - 1)


def edges(rows: list[dict[str, Any]], i: int, total: float, pad: float = 0.15) -> tuple[float, float]:
    """Where verse i's clip starts and ends (seconds): `pad` of air on each side, never past halfway to its neighbours,
    so a clip holds no sound of the verses beside it. `total` is the recording's length."""
    row = rows[i]
    before = rows[i - 1]["end"] if i else 0.0
    after = rows[i + 1]["start"] if i + 1 < len(rows) else total
    return max(row["start"] - pad, (before + row["start"]) / 2), min(row["end"] + pad, (row["end"] + after) / 2)
