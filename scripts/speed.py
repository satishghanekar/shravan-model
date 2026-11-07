"""How long the phone models take, on this Mac's CPU with 4 threads as an iPhone runs them: a CTC model on 2 and 5
seconds of sound, beside Whisper's encoder (always a 30-second block) and its decoder writing 25 tokens.

    python scripts/speed.py models/ctc-int8 models/v2-int8

Only the ratio means much: an iPhone took about 1.1 s for Whisper's encoder and 20 ms a token (2026-11-06), so the CTC
model's time there is about its share of Whisper's here. Writes one file to results/.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from shravan import config, results

THREADS = 4


def session(path: Path) -> ort.InferenceSession:
    o = ort.SessionOptions()
    o.intra_op_num_threads = THREADS
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def best_of(run: object, times: int = 5) -> float:
    """The quickest of a few runs, in ms (the first warms up)."""
    out = []
    for _ in range(times + 1):
        t = time.perf_counter()
        run()  # type: ignore[operator]
        out.append((time.perf_counter() - t) * 1000)
    return round(min(out[1:]), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ctc", help="a CTC model shrunk for the phone (a folder with model.onnx)")
    ap.add_argument("whisper", help="Whisper shrunk for the phone (encoder_model.onnx, decoder_model_merged.onnx)")
    args = ap.parse_args()
    cfg = config.load()
    rng = np.random.default_rng(0)

    ctc = session(Path(args.ctc) / "model.onnx")
    scores: dict[str, object] = {"threads": THREADS}
    for seconds in (2, 5):
        x = rng.normal(0, 1, (1, 16000 * seconds)).astype(np.float32)
        scores[f"ctc_{seconds}s_ms"] = best_of(lambda x=x: ctc.run(None, {"input_values": x}))

    enc = session(Path(args.whisper) / "encoder_model.onnx")
    feats = rng.normal(0, 1, (1, 80, 3000)).astype(np.float32)
    scores["whisper_encoder_ms"] = best_of(lambda: enc.run(None, {"input_features": feats}), 3)
    hidden = enc.run(None, {"input_features": feats})[0]
    dec = session(Path(args.whisper) / "decoder_model_merged.onnx")
    names = [i.name for i in dec.get_inputs() if i.name.startswith("past_key_values")]
    heads, size = 8, 64

    def write(tokens: int = 25) -> None:
        past: dict[str, np.ndarray] = {n: np.zeros((1, heads, 0, size), np.float32) for n in names}
        ids = np.array([[50258, 50344, 50359, 50363]], np.int64)
        for n in range(tokens):
            feed = {"input_ids": ids, "encoder_hidden_states": hidden, "use_cache_branch": np.array([n > 0])}
            res = dec.run(None, {**feed, **past})
            outs = dict(zip([o.name for o in dec.get_outputs()], res, strict=True))
            for k in names:
                if n == 0 or ".decoder." in k:
                    past[k] = outs[k.replace("past_key_values", "present")]
            ids = np.array([[int(outs["logits"][0, -1].argmax())]], np.int64)

    scores["whisper_25_tokens_ms"] = best_of(write, 3)
    whisper = float(scores["whisper_encoder_ms"]) + float(scores["whisper_25_tokens_ms"])  # type: ignore[arg-type]
    scores["ctc_5s_share_of_whisper"] = round(float(scores["ctc_5s_ms"]) / whisper, 3)  # type: ignore[arg-type]
    path = results.save("speed", args.ctc, cfg, scores)
    print(scores)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
