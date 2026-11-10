"""Training on the Mac's GPU, which shares its memory with everything else."""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import TrainerCallback

EVERY = 50  # steps between handing the memory back


class FreeMemory(TrainerCallback):  # type: ignore[misc]
    """Hands the GPU's cached memory back after each evaluation and every EVERY steps.

    PyTorch keeps the memory each new shape of batch used: the first training run grew to 13 GB after one evaluation
    and slowed from 2 to 30 seconds a step in swap, and the CTC run on the longer clips added 2026-11-08 filled its
    8.9 GB cap and stopped at step 1203 of 3000. Both training scripts use it."""

    def __init__(self) -> None:
        self.freed = 0

    def free(self) -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.freed += 1

    def on_evaluate(self, *args: Any, **kwargs: Any) -> None:
        self.free()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step % EVERY == 0:
            self.free()
