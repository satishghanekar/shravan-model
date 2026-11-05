"""One file per run in results/: what was run, on which texts and recordings, with which settings, and the scores."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from shravan.config import ROOT, Config


def save(kind: str, model: str, cfg: Config, scores: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    source = (cfg.texts / "SOURCE").read_text(encoding="utf-8")
    commit = re.search(r"^commit: (\S+)", source, re.M)
    now = datetime.datetime.now()
    name = f"{now:%Y-%m-%d-%H%M}-{kind}-{re.sub(r'[^a-z0-9]+', '-', model.lower()).strip('-')}.json"
    record = {
        "kind": kind,
        "model": model,
        "when": now.isoformat(timespec="seconds"),
        "texts_commit": commit.group(1) if commit else None,
        "recordings": (cfg.recordings).read_text(encoding="utf-8").splitlines()[1:],
        "split": {"test": list(cfg.heldout_hymns), "validation": list(cfg.validation_hymns)},
        "min_alignment_score": cfg.min_alignment_score,
        "training": cfg.training,
        "scores": scores,
    } | (extra or {})
    path = ROOT / "results" / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
