"""The settings read cleanly, and the rules that keep tests fair and audio private hold."""

from pathlib import Path

from shravan import config

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
    assert CFG.heldout_hymns
    assert all(1 <= h <= 67 for h in CFG.heldout_hymns)
    # enough left to train on
    assert len(CFG.heldout_hymns) < 67 / 2


def test_full_precision_on_the_mac():
    assert CFG.training["precision"] == "fp32"
