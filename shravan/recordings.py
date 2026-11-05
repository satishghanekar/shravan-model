"""The recordings listed in recordings.csv, and where each one's working files go in work/."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    file: str  # its name in the audio folder (config.yaml, audio:)
    date: str
    texts: tuple[str, ...]  # text ids, or "routine"
    who: str  # initials, or "published"
    heldout: bool
    notes: str

    @property
    def slug(self) -> str:
        """A safe name for its working files: "Sri Pavamaana Suktha - Copy 1.mp3" → "sri-pavamaana-suktha-copy-1"."""
        return re.sub(r"[^a-z0-9]+", "-", Path(self.file).stem.lower()).strip("-")


def load(path: Path) -> list[Recording]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Recording(
                file=row["file"],
                date=row["date"],
                texts=tuple(t.strip() for t in row["texts"].split(";") if t.strip()),
                who=row["who"],
                heldout=row["heldout"].strip().lower() == "yes",
                notes=row.get("notes", ""),
            )
            for row in csv.DictReader(f)
        ]


def find(recordings: list[Recording], name: str) -> Recording:
    """A recording by its file name or its slug."""
    for r in recordings:
        if name in (r.file, r.slug):
            return r
    raise KeyError(f"no recording named {name!r} in recordings.csv")
