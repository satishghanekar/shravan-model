"""Keep what another Mac needs to train in iCloud Drive (config.yaml, training_kit:); bring it back to a fresh clone.

    python scripts/training_kit.py             # this Mac → iCloud: the items in config.yaml (training_kit_items:)
    python scripts/training_kit.py --add checkpoints/ctc-new/checkpoint-3000   # and a stopped run's checkpoint too
    python scripts/training_kit.py --prune     # and take out of the kit what is no longer here
    python scripts/training_kit.py --restore   # iCloud → this clone, once iCloud has brought the folder down

The recordings are in iCloud already; this copies what is made from them and the checkpoint behind the site's model,
with a list of every file (shravan/training_kit.py). Saving only adds and updates files unless --prune is given; a
restore never deletes anything here. The folder's README.md says the same for whoever opens it in Finder.
"""

import argparse
import datetime
import subprocess
import sys

from shravan import config, env
from shravan import training_kit as kit
from shravan.config import ROOT


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def texts_commit() -> str:
    source = ROOT / "texts" / "SOURCE"
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    commit = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("commit:")), "?")
    return commit[:7]


def save_base_model(cfg: config.Config) -> str:
    """The starting model, once: its page names no licence and could go; training needs it to start at all."""
    target = cfg.training_kit / kit.BASE
    if (target / "config.json").exists():
        return "already kept"
    env.use_certifi()
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=cfg.model, local_dir=target)
    return "kept"


def save(cfg: config.Config, add: list[str], prune: bool) -> None:
    items = [*cfg.training_kit_items, *add]
    here = [i for i in items if (ROOT / i).exists()]
    for i in items:
        if i not in here:
            print(f"not here, left as it is in the kit: {i}")
    cfg.training_kit.mkdir(parents=True, exist_ok=True)

    listed, about = kit.read_list(cfg.training_kit)
    source = kit.files(ROOT, here)
    new = kit.to_copy(source, listed)
    print(f"copying {len(new)} files ({kit.gigabytes(source, new):.2f} GB) to {cfg.training_kit}")
    kit.copy(ROOT, cfg.training_kit, new)
    listed.update({p: source[p] for p in new})

    gone = kit.stale(source, listed, here)
    if gone and prune:
        for p in gone:
            (cfg.training_kit / p).unlink(missing_ok=True)
            del listed[p]
        print(f"removed {len(gone)} files no longer here")
    elif gone:
        print(f"{len(gone)} files in the kit are no longer here, such as {gone[0]}")
        print("  training reads every recording's clips, so if they were removed on purpose: --prune")

    print(f"the starting model: {save_base_model(cfg)}")
    about = {
        "date": datetime.date.today().isoformat(),
        "commit": git("rev-parse", "--short", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
        "texts": texts_commit(),
        "model": cfg.model,
        "items": sorted({*about.get("items", []), *here}),
    }
    kit.write_list(cfg.training_kit, listed, about)
    (cfg.training_kit / "README.md").write_text(kit.instructions(about), encoding="utf-8")
    print(f"saved: {len(listed)} files, {kit.gigabytes(listed, listed):.2f} GB in all")


def restore(cfg: config.Config) -> None:
    listed, about = kit.read_list(cfg.training_kit)
    if not listed:
        sys.exit(
            f"no training kit at {cfg.training_kit}\n"
            "  In Finder: iCloud Drive → development → shravan-model, right-click, Keep Downloaded; then again."
        )
    wanted = {p: f for p, f in listed.items() if not kit.under(p, kit.BASE)}
    absent = kit.missing(cfg.training_kit, wanted)
    if absent:
        sys.exit(
            f"{len(absent)} of {len(wanted)} files haven't come down from iCloud yet, such as {absent[0]}\n"
            "  Keep Downloaded on the shravan-model folder in Finder, wait until it has no cloud icon, and run again."
        )

    here = kit.files(ROOT, about.get("items", []))
    new = kit.to_copy(wanted, here)
    print(f"bringing {len(new)} files ({kit.gigabytes(wanted, new):.2f} GB) from the kit of {about.get('date')}")
    kit.copy(cfg.training_kit, ROOT, new)
    short = kit.missing(ROOT, {p: wanted[p] for p in new})
    if short:
        sys.exit(f"{len(short)} files came out short, such as {short[0]}: run the restore again")

    print(f"restored: {', '.join(about.get('items', []))}")
    if about.get("commit") and about["commit"] != git("rev-parse", "--short", "HEAD"):
        print(f"note: the kit was saved at commit {about['commit']}; this clone is at another")
    if not cfg.audio.exists():
        print(f"note: no recordings at {cfg.audio}; training needs only the clips, but align.py and cut.py need them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restore", action="store_true", help="bring the kit into this clone")
    parser.add_argument("--add", action="append", default=[], help="another folder to back up, relative to here")
    parser.add_argument("--prune", action="store_true", help="remove from the kit what is no longer here")
    args = parser.parse_args()
    cfg = config.load()
    if args.restore:
        restore(cfg)
    else:
        save(cfg, args.add, args.prune)


if __name__ == "__main__":
    main()
