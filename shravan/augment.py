"""Training clips made a little different each time they are used, so one chanter's takes aren't simply memorised."""

from __future__ import annotations

import random

import librosa
import numpy as np


def augment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random speed (±10%, which also shifts the pitch a little), volume and quiet noise."""
    rate = rng.uniform(0.9, 1.1)
    audio = librosa.resample(audio, orig_sr=16000, target_sr=int(16000 / rate))
    audio = audio * rng.uniform(0.6, 1.2)
    audio = audio + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 0.004), audio.shape)
    return audio.astype(np.float32)
