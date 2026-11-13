"""The iCloud backup copies what changed, removes only when asked, and a restore waits for a whole copy."""

import os
from pathlib import Path

from shravan import backup, config

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
    found = backup.files(tmp_path, ["work/clips", "notes.txt", "not-there"])
    assert sorted(found) == ["notes.txt", "work/clips/a/1.wav", "work/clips/a/manifest.jsonl"]
    assert found["work/clips/a/1.wav"] == backup.File("work/clips/a/1.wav", 3, 1_700_000_000)


def test_only_new_or_changed_files_are_copied():
    old = {"a": backup.File("a", 3, 1), "b": backup.File("b", 3, 1)}
    new = {"a": backup.File("a", 3, 1), "b": backup.File("b", 4, 2), "c": backup.File("c", 1, 1)}
    assert backup.to_copy(new, old) == ["b", "c"]
    assert backup.to_copy(old, old) == []


def test_a_clip_cut_again_at_the_same_size_is_still_copied():
    assert backup.to_copy({"a": backup.File("a", 3, 2)}, {"a": backup.File("a", 3, 1)}) == ["a"]


def test_stale_files_are_only_those_under_the_items_backed_up():
    here = {"work/clips/a/1.wav": backup.File("work/clips/a/1.wav", 1, 1)}
    there = {
        **here,
        "work/clips/gone/1.wav": backup.File("work/clips/gone/1.wav", 1, 1),
        "checkpoints/x/best/model.safetensors": backup.File("checkpoints/x/best/model.safetensors", 1, 1),
    }
    assert backup.stale(here, there, ["work/clips"]) == ["work/clips/gone/1.wav"]
    # a folder named like an item's start is not under it
    assert not backup.under("work/clipsy/1.wav", "work/clips")


def test_the_list_reads_back_as_written(tmp_path):
    listed = {"work/a.wav": backup.File("work/a.wav", 5, 7)}
    backup.write_list(tmp_path, listed, {"date": "2026-11-13", "items": ["work"]})
    assert backup.read_list(tmp_path) == (listed, {"date": "2026-11-13", "items": ["work"]})
    assert not (tmp_path / (backup.LIST + ".new")).exists()
    assert backup.read_list(tmp_path / "nothing") == ({}, {})


def test_a_restore_names_files_not_yet_down_or_cut_short(tmp_path):
    write(tmp_path / "whole.wav", "abc")
    write(tmp_path / "short.wav", "a")
    listed = {p: backup.File(p, 3, 1) for p in ("whole.wav", "short.wav", "absent.wav")}
    assert backup.missing(tmp_path, listed) == ["absent.wav", "short.wav"]


def test_a_backup_and_restore_round_trip(tmp_path):
    mac, cloud, fresh = tmp_path / "mac", tmp_path / "cloud", tmp_path / "fresh"
    write(mac / "work/clips/a/1.wav", "clip")
    write(mac / "checkpoints/r/best/model.safetensors", "weights")
    items = ["work/clips", "checkpoints/r/best"]
    source = backup.files(mac, items)
    backup.copy(mac, cloud, backup.to_copy(source, {}))
    backup.write_list(cloud, source, {"items": items})

    listed, about = backup.read_list(cloud)
    assert backup.missing(cloud, listed) == []
    backup.copy(cloud, fresh, backup.to_copy(listed, backup.files(fresh, about["items"])))
    assert backup.files(fresh, items) == source
    # a second restore has nothing left to bring
    assert backup.to_copy(listed, backup.files(fresh, items)) == []


def test_the_instructions_say_how_to_restore():
    text = backup.instructions({"date": "2026-11-13", "items": ["work/clips"], "remote": "https://example/x.git"})
    assert "python scripts/backup.py --restore" in text
    assert "git clone https://example/x.git" in text
    assert "`work/clips`" in text


def test_the_backup_is_outside_the_repository_and_beside_the_recordings_not_in_them():
    assert ROOT not in CFG.backup.parents and CFG.backup != ROOT
    # every recording is looked for by name anywhere under the audio folder: clips in there would be found too
    assert CFG.audio not in CFG.backup.parents and CFG.backup != CFG.audio
    assert CFG.backup not in CFG.audio.parents


def test_everything_backed_up_is_kept_out_of_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for item in CFG.backup_keep:
        assert any(item.startswith(pattern) for pattern in ignored if pattern.endswith("/")), item
