"""A model shrunk for the phone (scripts/export.py), packed as the site loads it (scripts/to_site.py).

Cloudflare Pages serves no file over 25 MiB, so each file over PIECE bytes is cut into pieces, which the site joins
again; model.json lists them, with each piece's size and SHA-256, so a broken download is caught. The site also needs
what the Python side gets from the processor: the mel filters (to turn sound into the model's input), the token texts
(to turn its output into letters), and the tokens it starts with, ends on and never writes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def greedy(encoder: Any, decoder: Any, features: np.ndarray, manifest: dict[str, Any], most: int = 40) -> list[int]:
    """The tokens the model writes for one window, step by step as the site does it (onnxruntime sessions): the start
    tokens, then each time the likeliest token it may write, until the end token or `most` tokens."""
    import numpy as np

    hidden = encoder.run(None, {"input_features": features[None]})[0]
    layers = manifest["layers"]
    empty: np.ndarray = np.zeros((1, 8, 0, 64), dtype=np.float32)
    past = {
        f"past_key_values.{i}.{s}.{k}": empty
        for i in range(layers)
        for s in ("decoder", "encoder")
        for k in ("key", "value")
    }
    ids = list(manifest["start"])
    step = np.array(ids, dtype=np.int64)[None]
    out: list[int] = []
    names = [o.name for o in decoder.get_outputs()]
    for n in range(most):
        feed = {"input_ids": step, "encoder_hidden_states": hidden, "use_cache_branch": np.array([n > 0])} | past
        result = dict(zip(names, decoder.run(None, feed), strict=True))
        logits = result["logits"][0, -1].astype(np.float64)
        logits[manifest["never"]] = -np.inf
        if n == 0:
            logits[manifest["never_first"]] = -np.inf
        token = int(np.argmax(logits))
        if token == manifest["end"]:
            break
        out.append(token)
        for name in past:
            present = result[name.replace("past_key_values", "present")]
            # (the encoder's keys and values are made on the first step only, then kept)
            if n == 0 or ".decoder." in name:
                past[name] = present
        step = np.array([[token]], dtype=np.int64)
    return out
