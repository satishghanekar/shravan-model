"""Pack a model shrunk for the phone (scripts/export.py) into a folder the Nitya Upasana site loads.

    python scripts/to_site.py models/v2-int8 models/site/chant-v2

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
from transformers import WhisperProcessor

from shravan import package


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a folder from scripts/export.py, such as models/v2-int8")
    ap.add_argument("out", help="the folder to write, such as models/site/chant-v2")
    args = ap.parse_args()
    src, out = Path(args.model), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

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
    tokens = package.greedy(encoder, decoder, features, manifest)
    check: dict[str, object] = {
        "seconds": 3.0,
        "frames": {str(f): [round(float(v), 5) for v in features[:, f]] for f in frames},
        "tokens": tokens,
    }
    check["text"] = tok.decode(tokens)
    # (greedy wanders off at a near tie, so the site checks against these: the encoder's numbers, and the decoder fed
    # the tokens above one by one, with its choice and margin at each step)
    check |= package.forced(encoder, decoder, features, manifest, tokens)
    (out / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    for f in sorted(out.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
