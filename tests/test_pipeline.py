"""The pipeline's logic: verses as units, recordings, the metrics, and the aligner's window maths."""

from pathlib import Path

import pytest

from shravan import config, data, follow, metrics, recordings
from shravan.texts import load_all
from shravan.units import MAX_LETTERS, of, select, stanzas, sung, verses

CFG = config.load()
PAVAMANA = load_all(CFG.texts)["vedic-chant-pavamana-suktam"]


def test_pavamana_has_610_verse_units_with_clean_labels():
    units = verses(PAVAMANA)
    assert len(units) == 610
    first = units[0]
    assert first.id == "vedic-chant-pavamana-suktam/9.001.01"
    assert (first.hymn, first.verse) == (1, 1)
    assert first.label == "स्वादिष्ठया मदिष्ठया पवस्व सोम धारया इन्द्राय पातवे सुतः"
    assert len({u.id for u in units}) == 610


def test_every_recording_row_is_readable_and_names_no_one():
    for r in recordings.load(CFG.recordings):
        assert r.texts
        # initials (two or three letters) or "published": never a name
        assert r.who == "published" or (r.who.isalpha() and r.who.isupper() and len(r.who) <= 3)


def test_a_recording_has_a_safe_slug():
    r = recordings.Recording("Sri Pavamaana Suktha - Copy 1.mp3", "", ("x",), ("",), "published", False, "")
    assert r.slug == "sri-pavamaana-suktha-copy-1"


LALITA = load_all(CFG.texts)["vedic-chant-lalita-sahasranama"]


def test_any_text_has_a_unit_per_block_keyed_by_number_or_kind():
    units = stanzas(LALITA)
    keys = [u.key for u in units]
    blocks = list(dict.fromkeys(u.block for u in units))
    # the Nyasa, four Dhyanam verses and the opening line said once, 183 stanzas, the closing line
    assert blocks[:7] == ["once-1", "once-2", "once-3", "once-4", "once-5", "once-6", "1"]
    assert blocks[-2:] == ["183", "once-7"]
    assert len(blocks) == 190 and len(set(keys)) == len(keys)
    assert of(LALITA) == units
    # Pavamana's units are still its referenced verses
    assert of(PAVAMANA) == verses(PAVAMANA)


def test_a_recording_names_the_blocks_it_chants_in_order():
    picked = select(stanzas(LALITA), "132..183 once-7 once-2..once-5")
    assert [u.key for u in picked][:2] == ["132", "133"]
    assert list(dict.fromkeys(u.block for u in picked))[-6:] == [
        "183",
        "once-7",
        "once-2",
        "once-3",
        "once-4",
        "once-5",
    ]
    with pytest.raises(KeyError):
        select(stanzas(LALITA), "200")
    with pytest.raises(ValueError):
        select(stanzas(LALITA), "183..132")


def test_a_block_too_long_for_one_clip_is_cut_into_pieces_of_a_few_lines():
    rudra = stanzas(load_all(CFG.texts)["vedic-chant-shree-rudra"])
    first = [u for u in rudra if u.block == "1"]
    # the first anuvaka, about two minutes chanted: in pieces, each short enough for a clip, its lines all kept
    assert len(first) > 3 and all(len(u.label) <= MAX_LETTERS * 1.5 for u in first)
    # in even pieces, never a long one and a scrap
    assert min(len(u.label) for u in first) > MAX_LETTERS / 2
    assert [u.key for u in first][:2] == ["1.1", "1.2"]
    assert select(rudra, "1") == first and select(rudra, "1.2")[0] == first[1]
    # short verses stay whole
    assert all(u.key == u.block for u in stanzas(load_all(CFG.texts)["vedic-chant-durga-suktam"]))


def test_an_aarti_is_chanted_with_its_dhrupad_after_every_stanza():
    every = load_all(CFG.texts)
    assert [u.key for u in sung(every["aarti-sukhkarta"])] == ["1", "dhrupad-1", "2", "dhrupad-1", "3", "dhrupad-1"]
    # refrain-before-first: the dhrupad first of all
    assert [u.key for u in sung(every["aarti-jaydeva-jaydeva-samartha"])][:3] == ["dhrupad-1", "1", "dhrupad-1"]
    # a text without a refrain after every verse: each block once, in order
    assert sung(every["vedic-chant-purusha-suktam"]) == of(every["vedic-chant-purusha-suktam"])


def test_the_recordings_list_reads_the_parts_of_a_text():
    lalita = next(r for r in recordings.load(CFG.recordings) if r.texts == ("vedic-chant-lalita-sahasranama",))
    assert lalita.parts == ("132..183 once-7 once-2..once-5",)
    assert lalita.heldout and lalita.who == "SJ"
    # every recording's parts name real blocks
    every = load_all(CFG.texts)
    for r in recordings.load(CFG.recordings):
        for t, p in zip(r.texts, r.parts, strict=True):
            assert select(of(every[t]), p) if p else of(every[t])


def test_every_recording_has_its_own_working_files():
    # the slug names its audio, alignment and clips in work/: two recordings sharing one would mix their clips (a new
    # take with the old one's name, say)
    slugs = [r.slug for r in recordings.load(CFG.recordings)]
    assert len(slugs) == len(set(slugs))


def test_a_text_held_out_for_its_voice_is_never_trained_on():
    # Pavamana is split by hymn, so its held-out hymns protect it; any other text a held-out recording chants (the
    # Lalita Sahasranama, the Purusha Suktam) must be in no recording that is trained on
    rows = recordings.load(CFG.recordings)
    tested = {t for r in rows if r.heldout for t in r.texts} - {"vedic-chant-pavamana-suktam"}
    trained = {t for r in rows if not r.heldout for t in r.texts}
    assert tested and not tested & trained


def test_training_can_take_one_kind_of_recording_and_weighs_the_family_more():
    def clip(n: int, rec: str) -> data.Clip:
        return data.Clip(str(n), Path(f"{n}.wav"), "क", 0, "train", 1.0, rec)

    train = [clip(1, "published"), clip(2, "sg"), clip(3, "sg")]
    assert [c.id for c in data.weighed(train, {"sg"}, 3)] == ["1", "2", "3", "2", "3", "2", "3"]
    assert [c.id for c in data.weighed(train, {"sg"}, 3, "published")] == ["1"]
    assert [c.id for c in data.weighed(train, {"sg"}, 2, "family")] == ["2", "3", "2", "3"]
    with pytest.raises(ValueError):
        data.weighed(train, {"sg"}, 3, "temple")


def test_following_finds_where_the_heard_words_end():
    labels = ["स्वादिष्ठया मदिष्ठया पवस्व", "इन्द्राय पातवे सुतः", "रक्षोहा विश्वचर्षणिः"]
    # the end of one verse and the start of the next: the chanting is in the next
    assert follow.where("पातवे सुतः रक्षोहा विश्व", labels) == 2
    assert follow.where("मदिष्ठया पव", labels) == 0
    assert follow.where("", labels) == -1
    assert list(follow.nearby(0, 10)) == [0, 1, 2, 3]
    assert list(follow.nearby(8, 10)) == [7, 8, 9]


def test_letters_wrong_ignores_marks_and_dandas_in_what_was_heard():
    assert metrics.letters_wrong(["सोम धारया"], ["सोम॒ धार॑या ।"]) == 0
    assert metrics.letters_wrong(["सोम"], ["सम"]) > 0


def test_verse_found_picks_the_nearest_label():
    candidates = ["स्वादिष्ठया मदिष्ठया", "इन्द्राय पातवे सुतः", "रक्षोहा विश्वचर्षणिः"]
    assert metrics.closest("इन्द्राय पातवे सुत", candidates) == 1
    assert metrics.verses_found([1, 2], ["इन्द्राय पातवे", "रक्षोहा"], candidates) == 1.0


def test_align_window_places_units_in_order():
    torch = pytest.importorskip("torch")  # CI runs without PyTorch; the Mac runs this
    pytest.importorskip("torchaudio")
    from shravan.align import align_window

    # a toy emission: 3 letters (1, 2, 3), blank 0, wildcard last (4); unit A is letter 1 then 2, unit B is letter 3
    frames = [4, 4, 1, 1, 0, 2, 2, 0, 0, 3, 3, 4]
    emission = torch.full((len(frames), 5), -10.0)
    for i, k in enumerate(frames):
        emission[i, k] = 0.0
    placed = align_window(emission, [[1, 2], [3]])
    (a_start, a_end, a_score), (b_start, b_end, b_score) = placed
    assert a_start <= a_end < b_start <= b_end
    assert (a_start, b_start) == (2, 9)
    assert a_score > 0.9 and b_score > 0.9


def test_a_model_file_is_cut_into_pieces_the_site_can_serve_and_joined_back(tmp_path):
    from shravan import package

    src = tmp_path / "model.onnx"
    src.write_bytes(bytes(range(256)) * 200_000)  # 51.2 MB
    pieces = package.split(src, tmp_path, "m.onnx")
    assert [p["file"] for p in pieces] == ["m.onnx", "m.onnx.2", "m.onnx.3"]
    assert all(int(str(p["bytes"])) <= package.PIECE < 25 * 2**20 for p in pieces)
    assert package.join(tmp_path, pieces) == src.read_bytes()
    (tmp_path / "m.onnx.2").write_bytes(b"broken")
    with pytest.raises(ValueError):
        package.join(tmp_path, pieces)


def test_the_check_sound_is_the_same_every_time():
    pytest.importorskip("numpy")  # (not installed in CI)
    from shravan import package

    a, b = package.tone(), package.tone()
    assert a.shape == (48000,) and (a == b).all() and abs(a).max() < 0.6


def test_training_hands_the_gpu_memory_back_every_few_steps_in_both_scripts():
    pytest.importorskip("torch")  # (not installed in CI)
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from shravan.gpu import EVERY, FreeMemory

    f = FreeMemory()
    for step in range(1, 3 * EVERY + 1):
        f.on_step_end(None, SimpleNamespace(global_step=step), None)
    f.on_evaluate()
    assert f.freed == 4
    root = Path(__file__).parent.parent / "scripts"
    assert all("FreeMemory()" in (root / s).read_text() for s in ("train.py", "train_ctc.py"))


def test_a_recording_is_found_by_its_name_in_any_folder_of_the_audio(tmp_path):
    # recordings.csv is public: a family recording is listed by its file name, never under a folder with a name
    (tmp_path / "Family" / "X").mkdir(parents=True)
    (tmp_path / "Family" / "X" / "SG - a.m4a").write_bytes(b"")
    (tmp_path / "Professional").mkdir()
    (tmp_path / "Professional" / "b.mp3").write_bytes(b"")
    assert recordings.locate(tmp_path, "SG - a.m4a") == tmp_path / "Family" / "X" / "SG - a.m4a"
    assert recordings.locate(tmp_path, "Professional/b.mp3") == tmp_path / "Professional" / "b.mp3"
    with pytest.raises(FileNotFoundError):
        recordings.locate(tmp_path, "c.m4a")


def test_the_recordings_list_names_no_family_folder():
    # (the folders under Family/ are named after people: initials only in this public repository)
    assert all(not r.file.startswith("Family/") for r in recordings.load(CFG.recordings))
