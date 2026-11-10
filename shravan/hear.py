"""What a CTC model (wav2vec2) writes and scores for pieces of audio (16 kHz mono): on the Mac's GPU, or, for a model
exported for the phone (a folder with model.onnx), with ONNX Runtime on the CPU, as the phone will run it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from shravan import ctc, env, follow
from shravan.sound import normalise


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
            self.device = env.device()
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


def letters(model_name: str, audio: list[np.ndarray]) -> list[str]:
    """A CTC model's words for each piece."""
    model = Ctc(model_name)
    return [model.words(a) for a in audio]
