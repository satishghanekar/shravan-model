"""Cut a recording into one clip per verse from its alignment (scripts/align.py), with labels and splits.

    python scripts/cut.py "Sri Pavamaana Suktha - Copy 1.mp3"

Writes work/clips/<recording>/<verse>.wav and work/clips/<recording>/manifest.jsonl (one line per clip: id, file,
label, hymn, split, score). Verses aligned below min_alignment_score are left out and listed.
"""

import json
import sys

import soundfile as sf

from shravan import config, data, recordings
from shravan.split import split_of

MAX_SECONDS = 30.0  # the longest clip kept: the long blocks are cut to fit it (units.py); changing it changes the clips


def main() -> None:
    cfg = config.load()
    rec = recordings.find(recordings.load(cfg.recordings), sys.argv[1])
    rows = json.loads((cfg.work / "align" / f"{rec.slug}.json").read_text(encoding="utf-8"))
    audio, rate = sf.read(cfg.work / "audio" / f"{rec.slug}.wav", dtype="float32")
    out = cfg.work / "clips" / rec.slug
    out.mkdir(parents=True, exist_ok=True)

    # a family recording trained on is held to a higher bar: below it, most likely a fumble (config.yaml)
    family = rec.who != "published" and not rec.heldout
    bar = cfg.min_alignment_score_family if family else cfg.min_alignment_score
    kept, left_out, counts = [], [], {"train": 0, "validation": 0, "test": 0}
    used: set[str] = set()
    for i, row in enumerate(rows):
        if row["score"] < bar:
            left_out.append(row)
            continue
        start, end = data.edges(rows, i, len(audio) / rate)
        if end - start > MAX_SECONDS:
            left_out.append(row)
            continue
        # (a block chanted twice in one recording gets a second file, not the first one's)
        name = row["id"].split("/")[-1]
        while name + ".wav" in used:
            name += "-again"
        name += ".wav"
        used.add(name)
        sf.write(out / name, audio[int(start * rate) : int(end * rate)], rate)
        # a held-out recording is only ever tested on; the others are split by hymn
        split = "test" if rec.heldout else split_of(row["hymn"], cfg)
        counts[split] += 1
        kept.append(
            {"id": row["id"], "recording": rec.slug, "file": f"{rec.slug}/{name}", "label": row["label"]}
            | {"hymn": row["hymn"], "split": split, "score": row["score"], "seconds": round(end - start, 2)}
        )

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    minutes = {s: sum(k["seconds"] for k in kept if k["split"] == s) / 60 for s in counts}
    for s, n in counts.items():
        print(f"{s:>10}: {n} clips, {minutes[s]:.1f} min")
    print(f"left out: {len(left_out)} (alignment below {bar}, or longer than {MAX_SECONDS:g} s)")
    for r in left_out:
        print(f"  {r['id'].split('/')[1]} at {int(r['start'] // 60)}:{r['start'] % 60:04.1f}, score {r['score']}")


if __name__ == "__main__":
    main()
