"""What a model writes for pieces of audio (16 kHz mono), a batch at a time on the Mac's GPU."""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe(model_name: str, audio: list[np.ndarray], language: str, batch: int = 8) -> list[str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    out: list[str] = []
    for i in range(0, len(audio), batch):
        feats = processor(audio[i : i + batch], sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=225)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return out
