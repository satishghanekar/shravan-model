"""Which split a verse belongs to: training, validation (choosing the best checkpoint) or test (the final score).

A verse is in exactly one. Splitting by hymn, not by verse, keeps neighbouring verses (which share words and refrains)
from landing on both sides, which would flatter the test.
"""

from __future__ import annotations

from typing import Literal

from shravan.config import Config

Split = Literal["train", "validation", "test"]


def split_of(hymn: int, cfg: Config) -> Split:
    if hymn in cfg.heldout_hymns:
        return "test"
    if hymn in cfg.validation_hymns:
        return "validation"
    return "train"
