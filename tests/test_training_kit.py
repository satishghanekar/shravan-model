"""The training kit copies what changed, removes only when asked, and a restore waits for a whole copy."""

import os
from pathlib import Path

from shravan import config
from shravan import training_kit as kit

CFG = config.load()
ROOT = Path(__file__).resolve().parent.parent


def write(path: Path, text: str, mtime: int = 1_700_000_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_files_lists_folders_and_single_files_but_not_finders_own(tmp_path):
    write(tmp_path / "work/clips/a/1.wav", "one")
    write(tmp_path / "work/clips/a/manifest.jsonl", "{}")
    write(tmp_path / "work/clips/.DS_Store", "x")
    write(tmp_path / "notes.txt", "n")
    found = kit.files(tmp_path, ["work/clips", "notes.txt", "not-there"])
    assert sorted(found) == ["notes.txt", "work/clips/a/1.wav", "work/clips/a/manifest.jsonl"]
    assert found["work/clips/a/1.wav"] == kit.File("work/clips/a/1.wav", 3, 1_700_000_000)


def test_only_new_or_changed_files_are_copied():
    old = {"a": kit.File("a", 3, 1), "b": kit.File("b", 3, 1)}
    new = {"a": kit.File("a", 3, 1), "b": kit.File("b", 4, 2), "c": kit.File("c", 1, 1)}
    assert kit.to_copy(new, old) == ["b", "c"]
    assert kit.to_copy(old, old) == []


def test_a_clip_cut_again_at_the_same_size_is_still_copied():
    assert kit.to_copy({"a": kit.File("a", 3, 2)}, {"a": kit.File("a", 3, 1)}) == ["a"]


def test_stale_files_are_only_those_under_the_items_kept():
    here = {"work/clips/a/1.wav": kit.File("work/clips/a/1.wav", 1, 1)}
    there = {
        **here,
        "work/clips/gone/1.wav": kit.File("work/clips/gone/1.wav", 1, 1),
        "checkpoints/x/best/model.safetensors": kit.File("checkpoints/x/best/model.safetensors", 1, 1),
    }
    assert kit.stale(here, there, ["work/clips"]) == ["work/clips/gone/1.wav"]
    # a folder named like an item's start is not under it
    assert not kit.under("work/clipsy/1.wav", "work/clips")


def test_the_list_reads_back_as_written(tmp_path):
    listed = {"work/a.wav": kit.File("work/a.wav", 5, 7)}
    kit.write_list(tmp_path, listed, {"date": "2026-11-13", "items": ["work"]})
    assert kit.read_list(tmp_path) == (listed, {"date": "2026-11-13", "items": ["work"]})
    assert not (tmp_path / (kit.LIST + ".new")).exists()
    assert kit.read_list(tmp_path / "nothing") == ({}, {})


def test_a_restore_names_files_not_yet_down_or_cut_short(tmp_path):
    write(tmp_path / "whole.wav", "abc")
    write(tmp_path / "short.wav", "a")
    listed = {p: kit.File(p, 3, 1) for p in ("whole.wav", "short.wav", "absent.wav")}
    assert kit.missing(tmp_path, listed) == ["absent.wav", "short.wav"]


def test_a_save_and_restore_round_trip(tmp_path):
    mac, cloud, fresh = tmp_path / "mac", tmp_path / "cloud", tmp_path / "fresh"
    write(mac / "work/clips/a/1.wav", "clip")
    write(mac / "checkpoints/r/best/model.safetensors", "weights")
    items = ["work/clips", "checkpoints/r/best"]
    source = kit.files(mac, items)
    kit.copy(mac, cloud, kit.to_copy(source, {}))
    kit.write_list(cloud, source, {"items": items})

    listed, about = kit.read_list(cloud)
    assert kit.missing(cloud, listed) == []
    kit.copy(cloud, fresh, kit.to_copy(listed, kit.files(fresh, about["items"])))
    assert kit.files(fresh, items) == source
    # a second restore has nothing left to bring
    assert kit.to_copy(listed, kit.files(fresh, items)) == []


def test_the_instructions_say_how_to_restore():
    text = kit.instructions({"date": "2026-11-13", "items": ["work/clips"], "remote": "https://example/x.git"})
    assert "python scripts/training_kit.py restore" in text
    assert "git clone https://example/x.git" in text
    assert "`work/clips`" in text


def test_the_kit_is_outside_the_repository_and_beside_the_recordings_not_in_them():
    assert ROOT not in CFG.training_kit.parents and CFG.training_kit != ROOT
    # every recording is looked for by name anywhere under the audio folder: clips in there would be found too
    assert CFG.audio not in CFG.training_kit.parents and CFG.training_kit != CFG.audio
    assert CFG.training_kit not in CFG.audio.parents


def test_everything_in_the_kit_is_kept_out_of_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for item in CFG.training_kit_items:
        assert any(item.startswith(pattern) for pattern in ignored if pattern.endswith("/")), item


def test_the_recordings_and_the_kit_share_one_project_folder_in_icloud():
    # one folder to Keep Downloaded on another Mac, and nothing of the project scattered across iCloud Drive
    assert CFG.training_kit.parent == CFG.audio.parent
    assert CFG.training_kit.parent.name == ROOT.name == "shravan-model"


def test_the_command_always_says_which_way_it_copies():
    import subprocess
    import sys

    script = str(ROOT / "scripts" / "training_kit.py")
    bare = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert bare.returncode != 0 and "save" in bare.stderr and "restore" in bare.stderr
    helped = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
    assert helped.returncode == 0 and "this Mac → iCloud" in helped.stdout
