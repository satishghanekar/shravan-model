"""The label rules: only the letters that are said, in plain Devanagari."""

import re

import pytest

from shravan import config
from shravan.labels import plain
from shravan.texts import load_all

TEXTS = load_all(config.load().texts)


@pytest.mark.parametrize(
    "line, label",
    [
        # svara marks go, letters stay
        ("स्वादि॑ष्ठया॒ मदि॑ष्ठया॒ पव॑स्व सोम॒ धार॑या ।", "स्वादिष्ठया मदिष्ठया पवस्व सोम धारया"),
        # a verse number in parts, with its dandas
        ("इन्द्रा᳚य॒ पात॑वे सु॒तः ॥ ९.००१.०१ ॥", "इन्द्राय पातवे सुतः"),
        # the booklet's verse numbers with the closing danda missing or joined
        ("योना᳚वृ॒तस्य॑ सीदत ॥ ९.०३९.०६", "योनावृतस्य सीदत"),
        ("मदे᳚षु सर्व॒धा अ॑सि ॥ ९.०१८.०१॥", "मदेषु सर्वधा असि"),
        # a kampa digit with its marks
        ("अ॒भी३॒॑ ममघ्न्या᳚ उ॒त", "अभी ममघ्न्या उत"),
        ("स्वा॒ध्य१॒॑ः पव॑मानासो", "स्वाध्यः पवमानासो"),
        # the invisible joiner that keeps a halant shown
        ("म॒रुद्‌भ्यः॑", "मरुद्भ्यः"),
        # avagraha, and the site's line-break marks
        ("पयोऽथ᳚र्वाणो", "पयोथर्वाणो"),
        ("निढळावरी कर, ¦निढळावरी कर", "निढळावरी कर, निढळावरी कर"),
        # a whole verse number
        ("सुखकर्ता दुःखहर्ता ॥ १० ॥", "सुखकर्ता दुःखहर्ता"),
        # a long name's parts, shown by hyphens, are one word (and "-" is the aligner's blank)
        ("श्रीशिवा शिव-शक्त्यैक्य-रूपिणी ललिताम्बिका ।", "श्रीशिवा शिवशक्त्यैक्यरूपिणी ललिताम्बिका"),
    ],
)
def test_label(line, label):
    assert plain(line) == label


def test_no_label_keeps_a_mark_digit_or_danda():
    for text in TEXTS.values():
        for line in text.lines:
            label = plain(line.orig)
            assert not re.search(r"[॑-॔᳐-᳿०-९0-9।॥|ऽ¦‌‍]", label), (text.id, label)
            assert label == " ".join(label.split())
