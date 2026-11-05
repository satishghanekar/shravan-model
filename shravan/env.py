"""What the commands need from the machine before they start."""

from __future__ import annotations

import os


def use_certifi() -> None:
    """Python from python.org on a Mac has no root certificates of its own, so model downloads fail with
    CERTIFICATE_VERIFY_FAILED. Point it at certifi's, unless the person has chosen others."""
    if "SSL_CERT_FILE" not in os.environ:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
