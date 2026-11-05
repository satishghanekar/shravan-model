"""Shrink a trained model for the phone: ONNX with int8 weights, about 76 MB instead of 290 MB.

    python scripts/export.py checkpoints/base-v2-family/best models/v2-int8

Exports the model to ONNX (optimum), then stores each weight in one byte instead of four (dynamic quantization), the
decoder's two inner graphs (the first word, and the words after it) included. The folder is what the phone loads:
encoder_model.onnx, decoder_model_merged.onnx and the tokenizer's files. Test it like any model, on the CPU as the phone
will run it: scripts/evaluate.py and scripts/follow.py take the folder. Never in git (models/).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

SHRUNK = ("encoder_model.onnx", "decoder_model_merged.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a trained model's folder, such as checkpoints/base-v2-family/best")
    ap.add_argument("out", help="the folder to write, such as models/v2-int8")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(sys.executable).parent / "optimum-cli"
        task = "automatic-speech-recognition-with-past"  # (the decoder keeps what it has read, so each word is quick)
        subprocess.run([str(cli), "export", "onnx", "--model", args.model, "--task", task, tmp], check=True)
        for f in Path(tmp).iterdir():
            if f.suffix != ".onnx":
                shutil.copy(f, out / f.name)
        for name in SHRUNK:
            # EnableSubgraph: the merged decoder's weights sit in its inner graphs, left as they were without it
            quantize_dynamic(
                Path(tmp) / name, out / name, weight_type=QuantType.QUInt8, extra_options={"EnableSubgraph": True}
            )
            before, after = (Path(tmp) / name).stat().st_size, (out / name).stat().st_size
            print(f"{name}: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
