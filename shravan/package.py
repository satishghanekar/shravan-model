"""A model shrunk for the phone (scripts/export.py), packed as the site loads it (scripts/to_site.py).

Cloudflare Pages serves no file over 25 MiB, so each file over PIECE bytes is cut into pieces, which the site joins
again; model.json lists them, with each piece's size and SHA-256, so a broken download is caught.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # (numpy is loaded only where it is used: CI, with no numpy, still tests split and join)
    import numpy as np

PIECE = 20_000_000  # bytes: under Cloudflare Pages' 25 MiB a file, with room to spare


def split(src: Path, out: Path, name: str) -> list[dict[str, object]]:
    """src cut into pieces of at most PIECE bytes in out, named name, name.2, name.3, …; their sizes and hashes."""
    data = src.read_bytes()
    pieces = []
    for n, at in enumerate(range(0, len(data), PIECE), 1):
        chunk = data[at : at + PIECE]
        file = name if n == 1 else f"{name}.{n}"
        (out / file).write_bytes(chunk)
        pieces.append({"file": file, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    return pieces


def join(folder: Path, pieces: list[dict[str, object]]) -> bytes:
    """The pieces read back and joined, each checked against its hash (what the site does, in Python for the tests)."""
    out = b""
    for p in pieces:
        chunk = (folder / str(p["file"])).read_bytes()
        if hashlib.sha256(chunk).hexdigest() != p["sha256"]:
            raise ValueError(f"{p['file']} is not the piece model.json names")
        out += chunk
    return out


def tone(seconds: float = 3.0, rate: int = 16000) -> np.ndarray:
    """The check sound: a few tones rising and falling, the same on every machine (the site makes it the same way)."""
    import numpy as np

    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    wave = 0.3 * np.sin(2 * np.pi * (220 + 60 * t) * t) + 0.2 * np.sin(2 * np.pi * 440 * t) * np.cos(
        2 * np.pi * 0.5 * t
    )
    out: np.ndarray = wave.astype(np.float32)
    return out


def check_of(scores: np.ndarray, ids: list[int], text: str, seconds: float = 3.0) -> dict[str, object]:
    """What the site must get from the check sound: the choice at each step and the words they make, the scores at
    three steps number for number, and each output's mean over every step. A tone is no chanting, so every choice is
    the blank whatever is broken; the means see all the scores, so a difference anywhere shows (2026-11-10)."""
    steps = [0, len(ids) // 2, len(ids) - 1]
    return {
        "seconds": seconds,
        "ids": ids,
        "text": text,
        "scores": {str(t): [round(float(v), 4) for v in scores[t]] for t in steps},
        "means": [round(float(v), 4) for v in scores.mean(0)],
    }
