"""The pieces of a text that are aligned, cut into clips and split for training: one unit per verse.

A verse is one block of the text ([para 1], [2], [once], …); its label is the plain Devanagari of its lines joined by
spaces. Pavamana Suktam's verses carry their Rigveda reference in the English line (|| 9.001.01 ||), which gives each
one its hymn and verse number for the train / validation / test split (verses()). Any other text's units are its
blocks, each with a key: its number ("132"), or its kind and how many of that kind came before ("once-7")
(stanzas()). sung() puts an aarti's dhrupad back after every stanza. A recording of part of a text, or with a part said
twice, names its blocks in order (select()).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shravan.labels import plain
from shravan.texts import Line, Text

REF = re.compile(r"\|\|\s*(\d+)\.(\d{3})\.(\d{2})")


@dataclass(frozen=True)
class Unit:
    id: str  # the text's id and the verse's reference or block key: "vedic-chant-pavamana-suktam/9.001.01"
    hymn: int  # the Rigveda hymn (Pavamana Suktam); 0 for other texts
    verse: int
    label: str

    @property
    def key(self) -> str:
        return self.id.split("/", 1)[1]


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


def stanzas(text: Text) -> list[Unit]:
    """Every block of a text, in order, keyed by its number ("132") or its kind and count ("once-7")."""
    out: list[Unit] = []
    seen: dict[str, int] = {}
    for lines in blocks(text):
        name = lines[0].block
        number = re.fullmatch(r"(?:para )?(\d+)", name)
        if number:
            key = number.group(1)
        else:
            kind = name.split()[0]
            seen[kind] = seen.get(kind, 0) + 1
            key = f"{kind}-{seen[kind]}"
        label = " ".join(plain(ln.orig) for ln in lines).strip()
        if label:
            out.append(Unit(f"{text.id}/{key}", 0, int(number.group(1)) if number else 0, label))
    return out


def sung(text: Text) -> list[Unit]:
    """A text's units in the order they are chanted. With "refrain: every-verse" (an aarti) the dhrupad comes again
    after every stanza, and first of all with "refrain-before-first: yes", as the site shows it; otherwise each block
    once, in the file's order."""
    every = of(text)
    if text.header.get("refrain") != "every-verse":
        return every
    refrain = [u for u in every if u.key.startswith("dhrupad-")]
    out = (
        [u for u in every if u.key.startswith("once-") and every.index(u) < every.index(refrain[0])] if refrain else []
    )
    if text.header.get("refrain-before-first") == "yes":
        out += refrain
    for u in every:
        if u.key.isdigit():
            out += [u, *refrain]
    return out


def of(text: Text) -> list[Unit]:
    """A text's units: its referenced verses if it has them (Pavamana Suktam), else its blocks."""
    return verses(text) or stanzas(text)


def select(every: list[Unit], parts: str) -> list[Unit]:
    """The units a recording chants, in its order: keys and ranges separated by spaces,
    "132..183 once-7 once-2..once-5". A range runs in the text's order, both ends included."""
    where = {u.key: i for i, u in enumerate(every)}
    out: list[Unit] = []
    for part in parts.split():
        first, _, last = part.partition("..")
        for key in (first, last or first):
            if key not in where:
                raise KeyError(f"no block {key!r} in {every[0].id.split('/')[0] if every else 'the text'}")
        a, b = where[first], where[last or first]
        if b < a:
            raise ValueError(f"{part}: the range runs backwards")
        out += every[a : b + 1]
    return out
