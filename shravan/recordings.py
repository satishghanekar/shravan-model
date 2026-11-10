"""The recordings listed in recordings.csv, and where each one's working files go in work/."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recording:
    file: str  # its name in the audio folder (config.yaml, audio:), or under it (locate())
    date: str
    texts: tuple[str, ...]  # text ids
    parts: tuple[str, ...]  # for each text, the blocks chanted in order ("132..183 once-7"), or "" for all of it
    who: str  # initials, or "published"
    heldout: bool
    notes: str

    @property
    def slug(self) -> str:
        """A safe name for its working files: "Sri Pavamaana Suktha - Copy 1.mp3" → "sri-pavamaana-suktha-copy-1"."""
        return re.sub(r"[^a-z0-9]+", "-", Path(self.file).stem.lower()).strip("-")


def load(path: Path) -> list[Recording]:
    """The texts column lists text ids separated by ";", each optionally followed by ":" and the blocks chanted, for
    a recording of part of a text or with a part said again: "vedic-chant-lalita-sahasranama: 132..183 once-7"."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        named = [t.partition(":") for t in row["texts"].split(";") if t.strip()]
        out.append(
            Recording(
                file=row["file"],
                date=row["date"],
                texts=tuple(t.strip() for t, _, _ in named),
                parts=tuple(p.strip() for _, _, p in named),
                who=row["who"],
                heldout=row["heldout"].strip().lower() == "yes",
                notes=row.get("notes", ""),
            )
        )
    return out


def find(recordings: list[Recording], name: str) -> Recording:
    """A recording by its file name or its slug."""
    for r in recordings:
        if name in (r.file, r.slug):
            return r
    raise KeyError(f"no recording named {name!r} in recordings.csv")


def locate(audio: Path, file: str) -> Path:
    """Where a recording is: its path under the audio folder, or else the one file of that name in any folder under it.

    recordings.csv is public, and the family's folders are named after them (iCloud Drive, "Family/<first name>/"), so
    a family recording is listed by its file name alone, initials only (2026-11-09)."""
    if (audio / file).exists():
        return audio / file
    found = sorted(audio.rglob(Path(file).name))
    if len(found) != 1:
        raise FileNotFoundError(f"{file!r}: {len(found)} files of that name under {audio}")
    return found[0]
