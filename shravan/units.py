"""The pieces of a text that are aligned, cut into clips and split for training: one unit per verse.

A verse is one block of the text ([para 1], [2], …); its label is the plain Devanagari of its lines joined by spaces.
Pavamana Suktam's verses carry their Rigveda reference in the English line (|| 9.001.01 ||), which gives each one its
hymn and verse number for the train / validation / test split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shravan.labels import plain
from shravan.texts import Line, Text

REF = re.compile(r"\|\|\s*(\d+)\.(\d{3})\.(\d{2})")


@dataclass(frozen=True)
class Unit:
    id: str  # the text's id and the verse's reference: "vedic-chant-pavamana-suktam/9.001.01"
    hymn: int
    verse: int
    label: str


def blocks(text: Text) -> list[list[Line]]:
    """The text's lines grouped by block (a line's index starts again at 1 in each block)."""
    groups: list[list[Line]] = []
    for line in text.lines:
        if line.index == 1:
            groups.append([])
        groups[-1].append(line)
    return groups


def verses(text: Text) -> list[Unit]:
    """Every numbered verse of a text, in order: the blocks with a Rigveda reference. The prayers before and after the
    hymns have none and are left out."""
    units = []
    for lines in blocks(text):
        ref = next((m for m in (REF.search(ln.en) for ln in lines) if m), None)
        if ref:
            label = " ".join(plain(ln.orig) for ln in lines)
            uid = f"{text.id}/{ref.group(1)}.{ref.group(2)}.{ref.group(3)}"
            units.append(Unit(uid, int(ref.group(2)), int(ref.group(3)), label))
    return units
