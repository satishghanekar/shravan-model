"""The pipeline's logic: verses as units, recordings, the metrics, and the aligner's window maths."""

import pytest

from shravan import config, metrics, recordings
from shravan.texts import load_all
from shravan.units import verses

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
    r = recordings.Recording("Sri Pavamaana Suktha - Copy 1.mp3", "", ("x",), "published", False, "")
    assert r.slug == "sri-pavamaana-suktha-copy-1"


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
