"""What the commands need from the machine before they start."""

from __future__ import annotations

import os


def use_certifi() -> None:
    """Python from python.org on a Mac has no root certificates of its own, so model downloads fail with
    CERTIFICATE_VERIFY_FAILED. Point it at certifi's, unless the person has chosen others."""
    if "SSL_CERT_FILE" not in os.environ:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()


def cap_gpu_memory(share: float) -> None:
    """Keep PyTorch to `share` of the memory macOS recommends for the GPU (about 11 GB of a 16 GB Mac).

    Without a cap PyTorch may take up to 1.7 times that, and on a Mac the GPU shares memory with everything else:
    the first training run grew to 13 GB, went into swap and slowed from 2 to 30 seconds a step. With a cap it fails
    with a clear out-of-memory error instead. Read when the GPU is first used, so this must run before that; a value
    already set in the environment wins."""
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(share))
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(round(share * 0.8, 2)))
