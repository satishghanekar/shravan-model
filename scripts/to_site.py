"""Pack a model shrunk for the phone (scripts/export.py) into a folder the Nitya Upasana site loads.

    python scripts/to_site.py models/v2-int8 models/site/chant-v2
    python scripts/to_site.py models/ctc-v2-int8 models/site/chant-v3

A CTC model (wav2vec2) is one graph, model.onnx, in pieces; letters.json (each output's letter, "" for the tokens
that are not letters), model.json (the pieces, the sample rate, the blank, the word break, ms a step) and check.json
(the letter it picks at each step of the check sound, and the words they make). See ctc() below.

The folder is copied into the site as models/chant-v2/ (by the site's own rules and tests). It holds the encoder and
decoder in pieces under Cloudflare's 25 MiB a file (shravan/package.py), mel.bin (the 80 mel filters, float32,
201 × 80), tokens.json (each token's text, GPT-2 byte style), model.json (the pieces, the sound settings and the
tokens the model starts with, ends on and never writes) and check.json (what the site must get from a check sound).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, Wav2Vec2CTCTokenizer, WhisperProcessor

from shravan import ctc as letters_of
from shravan import package
from shravan.sound import normalise


def ctc(src: Path, out: Path) -> None:
    """A CTC model packed for the site. It hears raw 16 kHz sound, each clip brought to mean 0 and spread 1 as it was
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
    # (a tone is no chanting: every step is the blank. So the scores of a few steps too, which the site's own run of
    # the model must match, number for number, whatever it hears)
    steps = [0, len(ids) // 2, len(ids) - 1]
    check = {
        "seconds": 3.0,
        "ids": ids,
        "text": letters_of.decode(ids, vocab, int(manifest["blank"])),
        "scores": {str(t): [round(float(v), 4) for v in scores[t]] for t in steps},
    }
    (out / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a folder from scripts/export.py, such as models/v2-int8")
    ap.add_argument("out", help="the folder to write, such as models/site/chant-v2")
    args = ap.parse_args()
    src, out = Path(args.model), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if "Wav2Vec2ForCTC" in (AutoConfig.from_pretrained(src).architectures or []):
        ctc(src, out)
        for f in sorted(out.iterdir()):
            print(f"{f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")
        return

    processor = WhisperProcessor.from_pretrained(src)
    fe, tok = processor.feature_extractor, processor.tokenizer
    gen = json.loads((src / "generation_config.json").read_text())
    config = json.loads((src / "config.json").read_text())
    np.asarray(fe.mel_filters, dtype="<f4").tofile(out / "mel.bin")
    vocab = tok.get_vocab()
    texts = [""] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        texts[i] = t
    (out / "tokens.json").write_text(json.dumps(texts, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    manifest = {
        "name": out.name,
        "encoder": package.split(src / "encoder_model.onnx", out, "encoder.onnx"),
        "decoder": package.split(src / "decoder_model_merged.onnx", out, "decoder.onnx"),
        "sound": {
            "rate": fe.sampling_rate,
            "fft": fe.n_fft,
            "hop": fe.hop_length,
            "mels": fe.feature_size,
            "seconds": fe.chunk_length,
        },
        # <|startoftranscript|> <|sa|> <|transcribe|> <|notimestamps|>
        "start": [gen["decoder_start_token_id"], gen["lang_to_id"]["<|sa|>"], gen["task_to_id"]["transcribe"]]
        + [gen["no_timestamps_token_id"]],
        "end": gen["eos_token_id"],
        "never": gen["suppress_tokens"],
        "never_first": gen["begin_suppress_tokens"],
        "first_special": tok.convert_tokens_to_ids("<|endoftext|>"),
        # the decoder's cache: one key and value per layer, for itself and for the encoder
        "layers": config["decoder_layers"],
    }
    (out / "model.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # check.json: what the site must get from the check sound (package.tone), so it can test its own sound features and
    # decoding against these, number for number and token for token, with no recording in the site
    features = fe(package.tone(), sampling_rate=fe.sampling_rate, return_tensors="np").input_features[0]
    encoder = ort.InferenceSession(package.join(out, manifest["encoder"]))
    decoder = ort.InferenceSession(package.join(out, manifest["decoder"]))
    frames = [0, 1, 150, 299, 300, 2999]
    check = {
        "seconds": 3.0,
        "frames": {str(f): [round(float(v), 5) for v in features[:, f]] for f in frames},
        "tokens": package.greedy(encoder, decoder, features, manifest),
    }
    check["text"] = tok.decode(check["tokens"])
    (out / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
