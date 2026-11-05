"""Copy the chant texts from the Nitya Upasana site into texts/, so every training run uses a fixed, recorded copy.

    python scripts/sync_texts.py                       # from ../nitya-upasana
    python scripts/sync_texts.py /path/to/nitya-upasana

The texts are copied exactly as they are, svara marks and all; shravan/labels.py decides what the model learns.
texts/SOURCE records the site commit they came from, so a model's scores can always be traced to the words it was
trained on.
"""

import datetime
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the repository, one level up from scripts/


def git(site: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(site), *args], capture_output=True, text=True).stdout.strip()


def main() -> None:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "nitya-upasana").resolve()
    src = site / "content" / "texts"
    if not src.is_dir():
        sys.exit(f"no texts folder at {src}")

    dest = ROOT / "texts"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    files = sorted(src.glob("*.txt"))
    for f in files:
        shutil.copy2(f, dest / f.name)
    shutil.copy2(site / "content" / "library.json", dest / "library.json")

    dirty = git(site, "status", "--porcelain", "content/texts", "content/library.json")
    (dest / "SOURCE").write_text(
        f"site: {git(site, 'remote', 'get-url', 'origin')}\n"
        f"commit: {git(site, 'rev-parse', 'HEAD')}\n"
        f"branch: {git(site, 'rev-parse', '--abbrev-ref', 'HEAD')}\n"
        f"uncommitted changes: {'yes' if dirty else 'no'}\n"
        f"copied: {datetime.date.today().isoformat()}\n"
        f"texts: {len(files)}\n",
        encoding="utf-8",
    )
    print(f"copied {len(files)} texts and library.json from {site} ({git(site, 'rev-parse', '--short', 'HEAD')})")
    if dirty:
        print("warning: the site has uncommitted changes to its texts; this copy can't be traced to a commit")


if __name__ == "__main__":
    main()
