"""Pack a model shrunk for the phone (scripts/export.py) into a folder the Nitya Upasana site loads.

    python scripts/to_site.py models/ctc-v3-int8 models/site/chant-v4
    python scripts/to_site.py models/ctc-v4-q4 models/site/chant-v4-q4

The model is one graph, model.onnx, in pieces under Cloudflare's 25 MiB a file (shravan/package.py); letters.json (each
output's letter, "" for the tokens that are not letters), model.json (the pieces, the sample rate, the blank, the word
break, ms a step) and check.json (what the site must get from the check sound). The folder is copied into the site as
models/chant-vN/, by the site's own rules and tests.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import Wav2Vec2CTCTokenizer

from shravan import ctc as letters_of
from shravan import package
from shravan.sound import normalise


def pack(src: Path, out: Path) -> None:
    """A model packed for the site. It hears raw 16 kHz sound, each clip brought to mean 0 and spread 1 as it was
    trained, and writes one of its outputs every STEP ms: the site needs no mel filters, only each output's letter."""
    tok = Wav2Vec2CTCTokenizer.from_pretrained(src)
    vocab = tok.get_vocab()
    names = [""] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        names[i] = "" if t in letters_of.SPECIAL else t
    (out / "letters.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    config = json.loads((src / "config.json").read_text())
    stride = int(np.prod(config["conv_stride"]))
    manifest = {
        "name": out.name,
        "kind": "ctc",
        "model": package.split(src / "model.onnx", out, "model.onnx"),
        "rate": 16000,
        # (each output covers this many ms: the feature encoder's strides multiplied, 320 samples)
        "step_ms": 1000 * stride / 16000,
        "blank": vocab[tok.pad_token],
        "word": vocab[letters_of.WORD],
    }
    (out / "model.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sound = package.tone()
    x = normalise(sound)[None]
    session = ort.InferenceSession(package.join(out, manifest["model"]))
    scores = session.run(None, {"input_values": x})[0][0]
    ids = [int(i) for i in scores.argmax(-1)]
    check = package.check_of(scores, ids, letters_of.decode(ids, vocab, int(manifest["blank"])))
    (out / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a folder from scripts/export.py, such as models/ctc-v3-int8")
    ap.add_argument("out", help="the folder to write, such as models/site/chant-v4")
    args = ap.parse_args()
    src, out = Path(args.model), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pack(src, out)
    for f in sorted(out.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
