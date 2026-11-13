"""What another Mac needs to train, kept in iCloud Drive beside the recordings (scripts/backup.py).

The recordings are in iCloud already. What is made from them (each unit's timings, the clips and the 16 kHz copies in
work/) takes an hour or more to make again, and the model on the site (its checkpoint) can't be made bit for bit the
same once a library has moved on. A copy of both, with a list of every file and its size, lets a fresh clone train
within minutes, and lets the restore tell a copy iCloud hasn't finished bringing down from a whole one."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LIST = "files.json"  # the list of every file in the backup, written last, so a backup cut short is never taken as whole
BASE = "base-model"  # the starting model from Hugging Face, kept in case it is ever taken down; never restored
SKIP = {".DS_Store"}  # Finder's own files, which iCloud adds and removes as it likes


@dataclass(frozen=True)
class File:
    path: str  # relative to the folder it is in, with "/" on every machine
    size: int
    mtime: int  # whole seconds: copies and iCloud keep no finer


def files(root: Path, items: Iterable[str]) -> dict[str, File]:
    """Every file under each item (a folder or a file, relative to root), by its path relative to root."""
    out = {}
    for item in items:
        top = root / item
        found = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file()) if top.is_dir() else []
        for p in found:
            if p.name in SKIP:
                continue
            s = p.stat()
            rel = p.relative_to(root).as_posix()
            out[rel] = File(rel, s.st_size, int(s.st_mtime))
    return out


def under(path: str, item: str) -> bool:
    return path == item or path.startswith(item.rstrip("/") + "/")


def to_copy(source: dict[str, File], target: dict[str, File]) -> list[str]:
    """The files of source that target lacks or has in another form (another size or time: cut again, trained on)."""
    return sorted(p for p, f in source.items() if target.get(p) != f)


def stale(source: dict[str, File], target: dict[str, File], items: Iterable[str]) -> list[str]:
    """The files of target under items that source no longer has. Training reads every recording's clips in work/clips,
    listed in recordings.csv or not, so a recording taken out here must go from the backup too; but only when asked
    (--prune), since a backup from a Mac not yet restored would otherwise empty the backup."""
    items = list(items)
    return sorted(p for p in target if p not in source and any(under(p, i) for i in items))


def read_list(folder: Path) -> tuple[dict[str, File], dict[str, Any]]:
    """The backup's files and what it says about itself (when, from which commit, which items); empty if none."""
    path = folder / LIST
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {f["path"]: File(**f) for f in raw["files"]}, dict(raw["about"])


def write_list(folder: Path, listed: dict[str, File], about: dict[str, Any]) -> None:
    """Written to a second name and then renamed, so iCloud never carries a half-written list to the other Mac."""
    body = {"about": about, "files": [asdict(listed[p]) for p in sorted(listed)]}
    tmp = folder / (LIST + ".new")
    tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, folder / LIST)


def missing(folder: Path, listed: dict[str, File]) -> list[str]:
    """The listed files not in folder at their full size. iCloud brings a folder to a new Mac file by file, so a restore
    started too early would train on part of the clips without a word; it stops and names what hasn't come yet."""
    out = []
    for p, f in listed.items():
        try:
            size = (folder / p).stat().st_size
        except FileNotFoundError:
            out.append(p)
            continue
        if size != f.size:
            out.append(p)
    return sorted(out)


def copy(source: Path, target: Path, paths: Iterable[str]) -> None:
    """Each file copied with its time, so the next comparison sees it as the same file."""
    for p in paths:
        dest = target / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / p, dest)


def gigabytes(listed: dict[str, File], paths: Iterable[str]) -> float:
    return sum(listed[p].size for p in paths) / 1e9


def instructions(about: dict[str, Any]) -> str:
    """The README written into the backup folder, so whoever opens it in Finder knows what it is and how to use it."""
    items = "\n".join(f"- `{i}`" for i in about.get("items", []))
    return f"""# Shravan's training kit

Made by `python scripts/backup.py` in the shravan-model repository on {about.get("date", "?")}, from commit
`{about.get("commit", "?")}` (texts from the site's `{about.get("texts", "?")}`). Private: clips cut from the family's
recordings. Never copy this folder into a git repository.

## What is here

{items}
- `{BASE}/`: the starting model ({about.get("model", "?")}), kept in case Hugging Face ever loses it. Training
  downloads it by itself; only if that fails, set `model:` in `config.yaml` to this folder.
- `{LIST}`: every file and its size. The restore checks it, so it never starts on a copy iCloud hasn't finished
  bringing down.

The recordings themselves are in `recordings/`, beside this folder.

## Training on another Mac

1. In Finder, open iCloud Drive → `development`, right-click `shravan-model` (the recordings and this kit):
   **Keep Downloaded**. Wait until it no longer shows the cloud icon.
2. In Terminal:

   ```sh
   brew install ffmpeg python@3.12
   mkdir -p ~/development && cd ~/development
   git clone {about.get("remote", "<this repository's address>")}
   cd shravan-model
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
   python scripts/backup.py --restore
   ```

   The restore stops and says what is missing if iCloud is still bringing files down: wait, and run it again.
3. Train, as the repository's README says ("The current model", or the run you mean to make).

When the run is done on that Mac, run `python scripts/backup.py` there, so this folder has its clips and checkpoint.
To carry a stopped run on elsewhere, add its last checkpoint: `python scripts/backup.py --add
checkpoints/<run>/checkpoint-<step>`, then `--restore` and `train_ctc.py --name <run> --resume` on the other Mac.
"""
