"""The settings read cleanly, and the rules that keep tests fair and audio private hold."""

from pathlib import Path

from shravan import config
from shravan.split import split_of

CFG = config.load()
ROOT = Path(__file__).resolve().parent.parent


def test_the_audio_is_outside_the_repository():
    assert ROOT not in CFG.audio.parents


def test_the_work_folder_is_kept_out_of_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "work/" in ignored
    for pattern in ("*.mp3", "*.m4a", "*.wav", "*.safetensors", "*.onnx"):
        assert pattern in ignored


def test_heldout_hymns_are_a_real_part_of_pavamana():
    assert CFG.heldout_hymns and CFG.validation_hymns
    assert all(1 <= h <= 67 for h in (*CFG.heldout_hymns, *CFG.validation_hymns))
    # enough left to train on
    assert len(CFG.heldout_hymns) + len(CFG.validation_hymns) < 67 / 2


def test_no_hymn_is_both_validation_and_test():
    assert not set(CFG.heldout_hymns) & set(CFG.validation_hymns)


def test_every_hymn_has_exactly_one_split():
    seen = {split_of(h, CFG) for h in range(1, 68)}
    assert seen == {"train", "validation", "test"}
    assert all(split_of(h, CFG) == "test" for h in CFG.heldout_hymns)
    assert all(split_of(h, CFG) == "validation" for h in CFG.validation_hymns)


def test_full_precision_on_the_mac():
    assert CFG.training["precision"] == "fp32"
