"""How long a phone model takes, on this Mac's CPU with 4 threads as an iPhone runs it: on 2, 4 and 5 seconds of sound
(Listen hears 4).

    python scripts/speed.py models/ctc-v3-int8

Only the ratio between models means much: on an iPhone chant-v4's 4 s took 712 ms on the CPU (2026-11-09). Writes one
file to results/.
"""

import argparse
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort

from shravan import config, results

THREADS = 4


def best_of(run: Callable[[], object], times: int = 5) -> float:
    """The quickest of a few runs, in ms (the first warms up)."""
    out = []
    for _ in range(times + 1):
        t = time.perf_counter()
        run()
        out.append((time.perf_counter() - t) * 1000)
    return round(min(out[1:]), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a model shrunk for the phone (a folder with model.onnx)")
    args = ap.parse_args()
    cfg = config.load()
    rng = np.random.default_rng(0)

    o = ort.SessionOptions()
    o.intra_op_num_threads = THREADS
    session = ort.InferenceSession(str(Path(args.model) / "model.onnx"), o, providers=["CPUExecutionProvider"])
    scores: dict[str, object] = {"threads": THREADS}
    for seconds in (2, 4, 5):
        feed = {"input_values": rng.normal(0, 1, (1, 16000 * seconds)).astype(np.float32)}
        scores[f"ctc_{seconds}s_ms"] = best_of(partial(session.run, None, feed))
    path = results.save("speed", args.model, cfg, scores)
    print(scores)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
