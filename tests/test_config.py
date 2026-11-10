"""The settings read cleanly, and the rules that keep tests fair and audio private hold."""

import os
from pathlib import Path

import pytest

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


def test_training_fits_a_16_gb_mac():
    t = CFG.training
    # a batch of 8 needed over 13 GB without recomputing activations: checkpointing stays on unless it shrinks
    assert t["checkpointing"] or t["batch_size"] <= 2
    # every run learns from the same number of clips per update, so their runs compare
    assert t["batch_size"] * t["accumulate"] == 8
    # above 1 PyTorch may take more than macOS recommends for the GPU, and the run goes into swap
    assert 0 < t["gpu_memory"] <= 1


def test_a_family_recording_trained_on_is_held_to_a_higher_bar():
    assert CFG.min_alignment_score <= CFG.min_alignment_score_family < 0.75


def test_family_recordings_weigh_more_but_not_overwhelmingly():
    assert 1 <= CFG.training["family_repeat"] <= 5


def test_gpu_memory_cap(monkeypatch):
    from shravan import env

    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    env.cap_gpu_memory(0.75)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.75"
    # PyTorch refuses to start if the low mark isn't below the high one
    assert float(os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]) < 0.75
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    env.cap_gpu_memory(0.75)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.5"


def test_a_setting_can_be_changed_for_one_comparison_run():
    assert config.overrides(["sort_by_length=false", "pad_to_seconds=4"]) == {
        "sort_by_length": False,
        "pad_to_seconds": 4,
    }
    with pytest.raises(ValueError):
        config.overrides(["sort_by_length"])


def test_family_validation_is_a_small_part_of_the_family_clips():
    assert 0 < CFG.training["family_validation"] <= 0.2
