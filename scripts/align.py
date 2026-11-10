"""Find where each verse starts and ends in a recording (shravan/align.py), and save the timings to work/align/.

    python scripts/align.py "Sri Pavamaana Suktha - Copy 1.mp3"
    python scripts/align.py "Lalita Sahasrnaam.m4a"

The recording must have its row in recordings.csv. Its audio is converted once to 16 kHz mono in work/audio/, and the
aligner's output is kept in work/emissions/, so a second run only aligns again.
"""

import json
import subprocess
import sys
import time

import soundfile as sf
import torch
import uroman
from torchaudio.pipelines import MMS_FA

from shravan import config, env, recordings, texts, units
from shravan.align import align, emissions, romanize_words


def main() -> None:
    env.use_certifi()
    cfg = config.load()
    rec = recordings.find(recordings.load(cfg.recordings), sys.argv[1])
    every = texts.load_all(cfg.texts)
    # each text in turn: the blocks the recording names, in its order, or the whole text as it is chanted
    verses = units.chanted(every, rec.texts, rec.parts)

    wav = cfg.work / "audio" / f"{rec.slug}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        source = recordings.locate(cfg.audio, rec.file)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "16000", str(wav)]
        subprocess.run(cmd, check=True)

    cache = cfg.work / "emissions" / f"{rec.slug}.pt"
    if cache.exists():
        saved = torch.load(cache)
        emission, times = saved["emission"], saved["times"]
    else:
        started = time.time()
        audio, rate = sf.read(wav, dtype="float32")
        where = env.device()
        model = MMS_FA.get_model().to(where)
        emission, times = emissions(model, torch.from_numpy(audio)[None], where, rate=rate)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"emission": emission, "times": times}, cache)
        print(f"listened to {times[-1] / 60:.0f} min of audio in {time.time() - started:.0f} s")

    dictionary = MMS_FA.get_dict()
    romanizer = uroman.Uroman()
    tokens = [[dictionary[c] for w in romanize_words(u.label, romanizer, dictionary) for c in w] for u in verses]

    started = time.time()
    placed = align(emission, times, verses, tokens)
    print(f"aligned {len(placed)} verses in {time.time() - started:.0f} s")

    out = cfg.work / "align" / f"{rec.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": p.unit.id, "hymn": p.unit.hymn, "verse": p.unit.verse, "label": p.unit.label}
        | {"start": round(p.start, 2), "end": round(p.end, 2), "score": round(p.score, 3)}
        for p in placed
    ]
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    scores = sorted(p.score for p in placed)
    lengths = sorted(p.end - p.start for p in placed)
    print(f"first verse at {placed[0].start:.1f} s, last ends at {placed[-1].end:.1f} s")
    print(f"scores: lowest {scores[0]:.2f}, median {scores[len(scores) // 2]:.2f}, highest {scores[-1]:.2f}")
    shortest, median, longest = lengths[0], lengths[len(lengths) // 2], lengths[-1]
    print(f"verse lengths: shortest {shortest:.1f} s, median {median:.1f} s, longest {longest:.1f} s")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
