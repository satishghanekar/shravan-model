"""Shrink a trained model for the phone: ONNX, one graph (model.onnx), about 108 MB instead of 380 MB.

    python scripts/export.py checkpoints/ctc-pro-all/best models/ctc-v3-int8
    python scripts/export.py checkpoints/ctc-pro-all/best models/ctc-v4-q4 --q4

Exports the model to ONNX (optimum), its weights worked out in advance first (fold), so its positional convolution is
shrunk too (122 MB without, 2026-11-07), then stores each weight in one byte instead of four (dynamic quantization).
The convolutions that turn raw sound into features are kept as they are: shrunk too (95 MB), its letters wrong went
from 11.5% to 19.3% on Pavamana and 15.1% to 23.6% on SG's Purusha Suktam.

--q4: its matrix weights in 4 bits (MatMulNBits, blocks of 32) instead of 8, the convolutions as they were: 90 MB, for a
phone's GPU (WebGPU), where the 8-bit model is no quicker than on the CPU. On an iPhone (Safari 27, 2026-11-09) a 4 s
hearing took 233 ms on the GPU against 712 ms for the 8-bit model on the CPU, and it hears as well.

Test it like any model, on the CPU as the phone will run it: scripts/evaluate.py and scripts/follow.py take the folder.
Never in git (models/).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

NAME = "model.onnx"


def sound_layers(path: Path) -> list[str]:
    """The convolutions that hear the raw sound (wav2vec2's feature encoder), left unshrunk."""
    graph = onnx.load(str(path), load_external_data=False).graph
    return [n.name for n in graph.node if n.op_type == "Conv" and "feature_extractor" in n.name]


def fold(src: Path, out: Path) -> None:
    """The graph with every value that depends only on the weights worked out once and stored. wav2vec2's positional
    convolution keeps its weight as a size and a direction (weight norm), multiplied together as it runs, which the
    quantizer can't shrink (19 MB of the 122); folded, it is a plain weight like the others."""
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    o.optimized_model_filepath = str(out)
    ort.InferenceSession(str(src), o, providers=["CPUExecutionProvider"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="a trained model's folder, such as checkpoints/ctc-pro-all/best")
    ap.add_argument("out", help="the folder to write, such as models/ctc-v3-int8")
    ap.add_argument("--q4", action="store_true", help="its matrix weights in 4 bits, for a phone's GPU")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cli = Path(sys.executable).parent / "optimum-cli"
        task = "automatic-speech-recognition"
        subprocess.run([str(cli), "export", "onnx", "--model", args.model, "--task", task, tmp], check=True)
        for f in Path(tmp).iterdir():
            if f.suffix != ".onnx":
                shutil.copy(f, out / f.name)
        exported = Path(tmp) / NAME
        folded = Path(tmp) / "folded.onnx"
        fold(exported, folded)
        if args.q4:
            q = MatMulNBitsQuantizer(onnx.load(str(folded)), block_size=32, is_symmetric=True)
            q.process()
            q.model.save_model_to_file(str(out / NAME))
        else:
            quantize_dynamic(
                folded,
                out / NAME,
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                nodes_to_exclude=sound_layers(folded),
            )
        print(f"{NAME}: {exported.stat().st_size / 1e6:.0f} MB -> {(out / NAME).stat().st_size / 1e6:.0f} MB")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
