"""One way to bring sound to what the model heard in training, the same as the site's."""

import numpy as np

from shravan.sound import normalise


def test_normalise_gives_what_the_sites_normalize_gives() -> None:
    # js/core/hearing.js normalize() on the same six samples (node, 2026-11-10)
    site = [0.5601847171783447, -0.7842586040496826, 1.4564802646636963, 0.33611080050468445, 0.11203693598508835]
    site += [-1.6805541515350342]
    out = normalise(np.array([0.1, -0.2, 0.3, 0.05, 0, -0.4], dtype=np.float32))
    assert out.dtype == np.float32
    assert np.allclose(out, site, atol=1e-6)


def test_silence_stays_silence() -> None:
    assert not normalise(np.zeros(16000, dtype=np.float32)).any()


def test_every_copy_of_the_sum_is_gone() -> None:
    # (it was written out in four places; one changed alone would make training, testing and the phone disagree)
    from pathlib import Path

    root = Path(__file__).parent.parent
    for f in [*root.glob("shravan/*.py"), *root.glob("scripts/*.py")]:
        if f.name != "sound.py":
            assert "a.var() + 1e-7" not in f.read_text() and "sound.var() + 1e-7" not in f.read_text(), f.name
