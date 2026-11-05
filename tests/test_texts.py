"""The copied texts parse, and Pavamana Suktam has every verse the training split relies on."""

import re

import pytest

from shravan import config
from shravan.texts import TextError, load_all, parse

CFG = config.load()
TEXTS = load_all(CFG.texts)


def test_every_site_text_parses():
    # library.json lists the site's texts; every one must be here and readable
    assert len(TEXTS) >= 51
    assert "vedic-chant-pavamana-suktam" in TEXTS


def test_texts_came_from_a_recorded_site_commit():
    source = (CFG.texts / "SOURCE").read_text(encoding="utf-8")
    assert re.search(r"^commit: [0-9a-f]{40}$", source, re.M)
    assert "uncommitted changes: no" in source


def test_pavamana_has_all_610_verses_in_order():
    refs = []
    for line in TEXTS["vedic-chant-pavamana-suktam"].lines:
        m = re.search(r"\|\| 9\.(\d{3})\.(\d{2})", line.en)
        if m:
            refs.append((int(m.group(1)), int(m.group(2))))
    assert len(refs) == 610
    assert refs[0] == (1, 1) and refs[-1] == (67, 32)


def test_a_line_pair_is_original_then_english():
    t = parse("id: x\ntitle: क\ntitle-en: K\nlanguage: sa\n---\n[1]\nक ।\nKa |\n", "x")
    assert [(ln.block, ln.index, ln.orig, ln.en) for ln in t.lines] == [("1", 1, "क ।", "Ka |")]


@pytest.mark.parametrize(
    "body, message",
    [
        ("[1]\nKa |\n", "original line"),
        ("[1]\nक ।\nख ।\n", "English line"),
        ("क ।\nKa |\n", "outside a block"),
        ("[1]\nक ।\n\n", "no English line"),
    ],
)
def test_mistakes_are_named(body, message):
    with pytest.raises(TextError, match=message):
        parse("id: x\ntitle: क\ntitle-en: K\nlanguage: sa\n---\n" + body, "x")
