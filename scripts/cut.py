"""Cut a recording into one clip per verse from its alignment (scripts/align.py), with labels and splits.

    python scripts/cut.py "Sri Pavamaana Suktha - Copy 1.mp3"

Writes work/clips/<recording>/<verse>.wav and work/clips/<recording>/manifest.jsonl (one line per clip: id, file,
label, hymn, split, score). Verses aligned below min_alignment_score are left out and listed.
"""

import json
import sys

import soundfile as sf

from shravan import config, recordings
from shravan.split import split_of

PAD = 0.15  # seconds of air kept before and after a verse, never reaching into its neighbours
MAX_SECONDS = 30.0  # Whisper hears at most 30 s at a time


def main() -> None:
    cfg = config.load()
    rec = recordings.find(recordings.load(cfg.recordings), sys.argv[1])
    rows = json.loads((cfg.work / "align" / f"{rec.slug}.json").read_text(encoding="utf-8"))
    audio, rate = sf.read(cfg.work / "audio" / f"{rec.slug}.wav", dtype="float32")
    out = cfg.work / "clips" / rec.slug
    out.mkdir(parents=True, exist_ok=True)

    kept, left_out, counts = [], [], {"train": 0, "validation": 0, "test": 0}
    for i, row in enumerate(rows):
        if row["score"] < cfg.min_alignment_score:
            left_out.append(row)
            continue
        before = rows[i - 1]["end"] if i else 0.0
        after = rows[i + 1]["start"] if i + 1 < len(rows) else len(audio) / rate
        start = max(row["start"] - PAD, (before + row["start"]) / 2)
        end = min(row["end"] + PAD, (row["end"] + after) / 2)
        if end - start > MAX_SECONDS:
            left_out.append(row)
            continue
        name = row["id"].split("/")[-1] + ".wav"
        sf.write(out / name, audio[int(start * rate) : int(end * rate)], rate)
        split = split_of(row["hymn"], cfg)
        counts[split] += 1
        kept.append(
            {"id": row["id"], "file": f"{rec.slug}/{name}", "label": row["label"]}
            | {"hymn": row["hymn"], "split": split, "score": row["score"], "seconds": round(end - start, 2)}
        )

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    minutes = {s: sum(k["seconds"] for k in kept if k["split"] == s) / 60 for s in counts}
    for s, n in counts.items():
        print(f"{s:>10}: {n} clips, {minutes[s]:.1f} min")
    print(f"left out: {len(left_out)} verses (alignment below {cfg.min_alignment_score})")


if __name__ == "__main__":
    main()
