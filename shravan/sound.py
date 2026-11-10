"""Sound as the chanting model hears it: each piece brought to mean 0 and spread 1, as it was trained (wav2vec2's
do_normalize), so a quiet phone and a loud one sound alike. One function for training, testing, packing and the site
(js/core/hearing.js normalize() does the same, and tests/test_sound.py checks the two agree on a few numbers)."""

from __future__ import annotations

import numpy as np


def normalise(audio: np.ndarray) -> np.ndarray:
    a: np.ndarray = audio.astype(np.float64)
    out: np.ndarray = ((a - a.mean()) / np.sqrt(a.var() + 1e-7)).astype(np.float32)
    return out
