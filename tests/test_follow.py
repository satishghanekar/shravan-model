"""shravan/follow.py against what Listen's own code answers (tests/fixtures/listen.json, made by listen.mjs from the
site's js/core/align.js and js/core/follow.js), so the release target is measured by the app's own method."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shravan import config, follow, units
from shravan.texts import load_all

LISTEN = json.loads((Path(__file__).parent / "fixtures" / "listen.json").read_text(encoding="utf-8"))
S = follow.Settings()
DOC = follow.Doc(LISTEN["doc"]["text"], LISTEN["doc"]["ends"], LISTEN["doc"]["refrain"])


def test_letters_are_the_sites() -> None:
    for case in LISTEN["letters"]:
        assert follow.letters(case["line"]) == case["expected"]


def test_a_doc_is_its_lines_letters_end_to_end() -> None:
    lines = [("स्वादि॒ष्ठया॒ मदि॑ष्ठया॒", False), ("पव॑स्व सोम॒ धार॑या ।", False), ("धृपद ध्रुव पद", True)]
    lines += [("इन्द्रा॑य॒ पात॑वे सु॒तः ॥ ९.००१.०१ ॥", False), ("र॒क्षो॒हा वि॒श्वच॑र्षणिः", False)]
    assert follow.make_doc(lines) == DOC
    assert [follow.line_at(DOC, i) for i in (0, 18, 19, 42, 43, 78)] == [0, 0, 1, 2, 3, 4]


@pytest.mark.parametrize("case", LISTEN["align"], ids=lambda c: f"{len(c['ids'])}-letters")
def test_alignment_is_the_sites(case: dict) -> None:
    outputs = case["outputs"]
    scores = np.array(case["scores"], dtype=np.float32).reshape(-1, outputs)
    lp = follow.log_probs(scores)
    assert np.allclose(lp.ravel(), case["log_probs"], atol=1e-6)
    # (lined up on the site's own log-probabilities, so a last-digit difference can't change the path)
    a = follow.align(np.array(case["log_probs"], dtype=np.float32).reshape(-1, outputs), case["ids"], 0, 1)
    want = case["expected"]
    assert a is not None
    assert (a.end, a.step, a.letters) == (want["end"], want["step"], want["letters"])
    assert a.fit == pytest.approx(want["fit"], abs=1e-9)


def test_nothing_to_line_up_with() -> None:
    assert follow.align(np.zeros((4, 3), dtype=np.float32), [], 0, 1) is None
    assert follow.align(np.zeros((0, 3), dtype=np.float32), [2], 0, 1) is None


def test_a_stretch_as_the_models_outputs() -> None:
    t = LISTEN["target"]
    assert follow.target(t["letters"], t["ids"], t["from"]) == (t["expected"]["ids"], t["expected"]["at"])


@pytest.mark.parametrize("case", LISTEN["match"], ids=lambda c: c["heard"])
def test_matching_letters_is_the_sites(case: dict) -> None:
    want = case["expected"]
    assert follow.match(case["heard"], DOC.text, case["from"], case["to"]) == (want["end"], want["cost"])


@pytest.mark.parametrize("case", LISTEN["locate"], ids=lambda c: c["heard"])
def test_locating_is_the_sites(case: dict) -> None:
    here = (case["here"]["doc"], case["here"]["line"]) if case["here"] else None
    found = follow.locate(case["heard"], [DOC], here, S)
    want = case["expected"]
    if want is None:
        assert found is None
    else:
        assert found is not None
        assert (found.doc, found.end, found.held, found.how) == (want["doc"], want["end"], want["held"], want["scope"])


def scores_for(text: str, m: follow.Model, steps_each: int = 3) -> np.ndarray:
    """Made-up scores as the model would give them for chanting `text`: each letter sure for a few steps, a blank
    between."""
    rows = []
    for ch in text:
        for k in (m.ids[ch],) * steps_each + (m.blank,):
            row = np.zeros(len(m.names), dtype=np.float32)
            row[k] = 8.0
            rows.append(row)
    return np.stack(rows)


def test_one_hearing_follows_near_the_mark_and_finds_a_lost_place() -> None:
    every = sorted(set(DOC.text))
    names = ["", "", *every]
    m = follow.Model({ch: i for i, ch in enumerate(names) if ch}, names, 0, 1)
    heard = DOC.text[DOC.ends[2] : DOC.ends[3]]  # the unit after the dhrupad
    scores = scores_for(heard, m)
    near = follow.hear(scores, [DOC], (0, float(DOC.ends[2])), m, S)
    assert near == follow.Found(0, DOC.ends[3], False, "aligned")
    lost = follow.hear(scores, [DOC], None, m, S)
    assert lost is not None and lost.how == "any" and follow.line_at(DOC, lost.end - 1) == 3
    # the dhrupad heard again, long after it is written: the mark waits
    held = follow.hear(scores_for(DOC.text[DOC.ends[1] : DOC.ends[2]], m), [DOC], (0, float(DOC.ends[4])), m, S)
    assert held is not None and held.held


def test_listens_settings_come_from_the_config() -> None:
    s = follow.Settings.of(config.load().listen)
    assert (s.window, s.fit, s.back, s.forth) == (4.0, -0.6, 60, 250)


TEXTS = load_all(config.load().texts)


def test_every_text_is_laid_out_as_the_site_lays_it_out_for_listen() -> None:
    # (made from the same copy of the texts: after a sync, make listen.json again, as listen.mjs says)
    assert LISTEN["texts_commit"] in (config.load().texts / "SOURCE").read_text(encoding="utf-8")
    for tid, digest in LISTEN["docs"].items():
        doc = follow.book([TEXTS[tid]]).docs[0]
        flags = "".join("1" if r else "0" for r in doc.refrain)
        assert hashlib.sha256(f"{doc.text}|{flags}".encode()).hexdigest() == digest, tid


def rows_of(units: list) -> list[dict]:
    """Units as aligned in a recording, each 5 s long."""
    return [{"id": u.id, "start": 5.0 * i, "end": 5.0 * i + 5} for i, u in enumerate(units)]


def test_a_recording_of_two_aartis_is_one_book_and_the_dhrupad_repeats_are_known() -> None:
    ids = ("aarti-sukhkarta", "aarti-ghalin-lotangan")
    b = follow.book([TEXTS[t] for t in ids])
    rows = rows_of(units.chanted(TEXTS, ids, ("", "")))
    assert [b.doc_of(r["id"]) for r in rows][:7] == [0] * 6 + [1]
    assert [b.is_refrain(r["id"]) for r in rows[:4]] == [False, True, False, True]
    # each stanza's letters are where the site shows it
    start, end = b.span(rows[2]["id"]) or (0, 0)
    assert b.keys[0][follow.line_at(b.docs[0], start)] == "2" and end > start


def test_the_mark_moves_on_from_the_last_unit_and_waits_through_the_dhrupad() -> None:
    ids = ("aarti-sukhkarta",)
    b = follow.book([TEXTS["aarti-sukhkarta"]])
    rows = rows_of(units.chanted(TEXTS, ids, ("",)))  # 1, dhrupad, 2, dhrupad, 3, dhrupad
    s = follow.Settings()
    assert follow.mark_at(b, rows, 0, 2.0, s) is None
    end_1 = (b.span(rows[0]["id"]) or (0, 0))[1]
    # the dhrupad after stanza 1: the mark waits at the end of stanza 1
    assert follow.mark_at(b, rows, 1, 7.0, s) == (0, end_1)
    # stanza 2, 3 s in: it waited through the dhrupad, so it is still there
    assert follow.mark_at(b, rows, 2, 13.0, s) == (0, end_1)
    # in a text with no dhrupad, it moves on at the pace from where the last unit ended
    pur = follow.book([TEXTS["vedic-chant-purusha-suktam"]])
    prow = rows_of(units.chanted(TEXTS, ("vedic-chant-purusha-suktam",), ("",)))
    end_0 = (pur.span(prow[0]["id"]) or (0, 0))[1]
    assert follow.mark_at(pur, prow, 1, 8.0, s) == (0, end_0 + 3 * s.pace)


def test_a_place_found_is_right_only_in_the_unit_being_chanted() -> None:
    b = follow.book([TEXTS["aarti-sukhkarta"]])
    rows = rows_of(units.chanted(TEXTS, ("aarti-sukhkarta",), ("",)))
    stanza_2, dhrupad = rows[2]["id"], rows[1]["id"]
    s2 = b.span(stanza_2) or (0, 0)
    assert follow.right(b, stanza_2, follow.Found(0, s2[1], False, "aligned"))
    assert not follow.right(b, stanza_2, follow.Found(0, s2[0], False, "aligned"))  # the end of stanza 1
    assert not follow.right(b, stanza_2, follow.Found(0, -1, True, "near"))
    assert not follow.right(b, stanza_2, None)
    # the dhrupad said again: the mark held, or the dhrupad found anywhere it is shown
    assert follow.right(b, dhrupad, follow.Found(0, -1, True, "aligned"))
    last_dhrupad_end = b.docs[0].ends[-1]
    assert follow.right(b, dhrupad, follow.Found(0, last_dhrupad_end, False, "any"))
    assert not follow.right(b, dhrupad, follow.Found(0, s2[1], False, "any"))
