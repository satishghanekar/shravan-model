"""What the model learns to write for a line: its plain Devanagari words.

The app follows the chanting by matching what the model hears against the known lines, letter by letter, so the
label keeps exactly the letters that are said and drops everything that is only written: svara marks (॒ ॑ ᳚ and the
other Vedic signs), the kampa digits (अ॒भी३॒॑), verse numbers (॥ ९.००१.०१ ॥), dandas, avagraha, the invisible joiners,
the site's line-break marks (¦, //) and the hyphens that show a long name's parts (शिव-शक्त्यैक्य-रूपिणी is one word,
शिवशक्त्यैक्यरूपिणी). A change here changes every model: retrain the baseline after one.
"""

from __future__ import annotations

import re
import unicodedata

# Svara and other Vedic marks: the four in the Devanagari block, and the Vedic Extensions block
VEDIC_MARKS = re.compile(r"[॑-॔᳐-᳿]")
# A verse number between dandas, whole (॥ १० ॥) or in parts (॥ ९.००१.०१ ॥), the closing danda sometimes missing
VERSE_NUMBER = re.compile(r"[॥।|]+\s*[०-९0-9]+(?:\.[०-९0-9]+)*\s*(?:[॥।|]+)?")
# What is left of the writing that is not said
NOT_SAID = re.compile(r"[०-९0-9।॥|ऽ¦‌‍-]")


def plain(line: str) -> str:
    """An original line → the words the model should write, separated by single spaces."""
    t = unicodedata.normalize("NFC", line)
    t = t.replace(" // ", " ")
    t = VERSE_NUMBER.sub(" ", t)
    t = VEDIC_MARKS.sub("", t)
    t = NOT_SAID.sub("", t)
    return " ".join(t.split())
