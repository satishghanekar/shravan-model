"""What a model writes for pieces of audio (16 kHz mono), a batch at a time: on the Mac's GPU, or, for a model
exported for the phone (a folder with encoder_model.onnx), with ONNX Runtime on the CPU, as the phone will run it.
A CTC model (wav2vec2, its config's architectures say so) writes letter by letter instead (shravan/ctc.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor

from shravan import ctc, follow
from shravan.sound import normalise


def transcribe(model_name: str, audio: list[np.ndarray], language: str, batch: int = 8) -> list[str]:
    if is_ctc(model_name):
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


class Ctc:
    """A CTC model (wav2vec2) loaded once: on the GPU, or, shrunk for the phone (a folder with model.onnx), with ONNX
    Runtime on the CPU, as the phone will run it. Each piece is heard on its own (no padding, which changes a wav2vec2
    model's normalisation)."""

    def __init__(self, model_name: str) -> None:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        self.alpha: dict[str, int] = tokenizer.get_vocab()
        self.blank: int = self.alpha[tokenizer.pad_token]
        self.model: Any
        if (Path(model_name) / "model.onnx").exists():
            from optimum.onnxruntime import ORTModelForCTC

            self.device = "cpu"
            self.model = ORTModelForCTC.from_pretrained(model_name)
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device).eval()

    def scores(self, audio: np.ndarray) -> np.ndarray:
        """The model's scores for each 20 ms step of one piece: steps × outputs."""
        x = torch.from_numpy(normalise(audio))[None].to(self.device)
        with torch.inference_mode():
            out: np.ndarray = self.model(x).logits[0].float().cpu().numpy()
        return out

    def words(self, audio: np.ndarray) -> str:
        """Each step's likeliest letter, read as shravan/ctc.py decode does."""
        return ctc.decode([int(i) for i in self.scores(audio).argmax(-1)], self.alpha, self.blank)

    def follow_model(self) -> follow.Model:
        """What shravan/follow.py needs: each letter's output, each output's letter, the blank and the word break."""
        names = [""] * (max(self.alpha.values()) + 1)
        for ch, i in self.alpha.items():
            names[i] = "" if ch in ctc.SPECIAL else ch
        ids = {ch: i for i, ch in enumerate(names) if ch and ch != ctc.WORD}
        return follow.Model(ids, names, self.blank, self.alpha[ctc.WORD])


def is_ctc(model_name: str) -> bool:
    return "Wav2Vec2ForCTC" in (AutoConfig.from_pretrained(model_name).architectures or [])


def letters(model_name: str, audio: list[np.ndarray]) -> list[str]:
    """A CTC model's words for each piece."""
    model = Ctc(model_name)
    return [model.words(a) for a in audio]
