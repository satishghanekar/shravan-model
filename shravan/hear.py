"""What a model writes for pieces of audio (16 kHz mono), a batch at a time: on the Mac's GPU, or, for a model
exported for the phone (a folder with encoder_model.onnx), with ONNX Runtime on the CPU, as the phone will run it.
A CTC model (wav2vec2, its config's architectures say so) writes letter by letter instead (shravan/ctc.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor

from shravan import ctc


def transcribe(model_name: str, audio: list[np.ndarray], language: str, batch: int = 8) -> list[str]:
    if "Wav2Vec2ForCTC" in (AutoConfig.from_pretrained(model_name).architectures or []):
        return letters(model_name, audio)
    processor = WhisperProcessor.from_pretrained(model_name)
    model: Any
    if (Path(model_name) / "encoder_model.onnx").exists():
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        device = "cpu"
        # (decoder_model_merged.onnx: one decoder for the first word and those after it)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, use_merged=True)
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    out: list[str] = []
    for i in range(0, len(audio), batch):
        feats = processor(audio[i : i + batch], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=225)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out


def letters(model_name: str, audio: list[np.ndarray]) -> list[str]:
    """A CTC model's words: each piece on its own (no padding, which changes a wav2vec2 model's normalisation), each
    20 ms step's likeliest letter, read as shravan/ctc.py decode does."""
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
    alpha = tokenizer.get_vocab()
    blank = alpha[tokenizer.pad_token]
    model: Any
    if (Path(model_name) / "model.onnx").exists():
        # shrunk for the phone: on the CPU, as the phone will run it
        from optimum.onnxruntime import ORTModelForCTC

        device = "cpu"
        model = ORTModelForCTC.from_pretrained(model_name)
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    out = []
    for a in audio:
        x = torch.from_numpy((a - a.mean()) / np.sqrt(a.var() + 1e-7)).float()[None].to(device)
        with torch.inference_mode():
            ids = model(x).logits.argmax(-1)[0].tolist()
        out.append(ctc.decode(ids, alpha, blank))
    return out
