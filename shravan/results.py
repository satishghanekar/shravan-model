"""One file per run in results/: what was run, on which texts and recordings, with which code, settings and libraries,
and the scores, so any score can be traced and run again."""

from __future__ import annotations

import contextlib
import datetime
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from shravan.config import ROOT, Config

# the libraries a score depends on (a new version of one can change what a model hears or how it is shrunk)
LIBRARIES = ("torch", "torchaudio", "transformers", "onnxruntime", "onnx", "optimum", "optimum-onnx", "numpy")


def code() -> dict[str, Any]:
    """This repository's commit, and whether the run's code had changes not yet committed."""

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "uncommitted_changes": bool(git("status", "--porcelain", "--", "shravan", "scripts", "config.yaml")),
    }


def libraries() -> dict[str, str]:
    out = {}
    for name in LIBRARIES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[name] = metadata.version(name)
    return out


def save(
    kind: str,
    model: str,
    cfg: Config,
    scores: dict[str, Any],
    extra: dict[str, Any] | None = None,
    folder: Path = ROOT / "results",
) -> Path:
    source = (cfg.texts / "SOURCE").read_text(encoding="utf-8")
    commit = re.search(r"^commit: (\S+)", source, re.M)
    now = datetime.datetime.now()
    stem = f"{now:%Y-%m-%d-%H%M}-{kind}-{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}"
    # (two runs in one minute keep both files)
    path, n = folder / f"{stem}.json", 2
    while path.exists():
        path, n = folder / f"{stem}-{n}.json", n + 1
    record = {
        "kind": kind,
        "model": model,
        "when": now.isoformat(timespec="seconds"),
        "texts_commit": commit.group(1) if commit else None,
        "code": code(),
        "libraries": libraries(),
        "recordings": (cfg.recordings).read_text(encoding="utf-8").splitlines()[1:],
        "split": {"test": list(cfg.heldout_hymns), "validation": list(cfg.validation_hymns)},
        "min_alignment_score": cfg.min_alignment_score,
        "min_alignment_score_family": cfg.min_alignment_score_family,
        "training": cfg.training,
        "scores": scores,
    } | (extra or {})
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
