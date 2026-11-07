"""Shrink a trained model for the phone: ONNX with int8 weights, about 76 MB instead of 290 MB.

    python scripts/export.py checkpoints/base-v2-family/best models/v2-int8

Exports the model to ONNX (optimum), then stores each weight in one byte instead of four (dynamic quantization), the
decoder's two inner graphs (the first word, and the words after it) included. The folder is what the phone loads:
encoder_model.onnx, decoder_model_merged.onnx and the tokenizer's files. A CTC model (wav2vec2) is one graph,
model.onnx, about 95 MB instead of 380 MB. Test it like any model, on the CPU as the phone
will run it: scripts/evaluate.py and scripts/follow.py take the folder. Never in git (models/).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig

SHRUNK = ("encoder_model.onnx", "decoder_model_merged.onnx")
SHRUNK_CTC = ("model.onnx",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a trained model's folder, such as checkpoints/base-v2-family/best")
    ap.add_argument("out", help="the folder to write, such as models/v2-int8")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    is_ctc = "Wav2Vec2ForCTC" in (AutoConfig.from_pretrained(args.model).architectures or [])
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(sys.executable).parent / "optimum-cli"
        # (Whisper's decoder keeps what it has read, so each word is quick; a CTC model has no decoder)
        task = "automatic-speech-recognition" if is_ctc else "automatic-speech-recognition-with-past"
        subprocess.run([str(cli), "export", "onnx", "--model", args.model, "--task", task, tmp], check=True)
        for f in Path(tmp).iterdir():
            if f.suffix != ".onnx":
                shutil.copy(f, out / f.name)
        for name in SHRUNK_CTC if is_ctc else SHRUNK:
            # EnableSubgraph: the merged decoder's weights sit in its inner graphs, left as they were without it
            # (a CTC model: its matrix products only, nearly all its weights; its positional convolution's weight is
            # worked out from two others as it runs, which the quantizer can't shrink)
            quantize_dynamic(
                Path(tmp) / name,
                out / name,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm"] if is_ctc else None,
                extra_options={"EnableSubgraph": True},
            )
            before, after = (Path(tmp) / name).stat().st_size, (out / name).stat().st_size
            print(f"{name}: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
