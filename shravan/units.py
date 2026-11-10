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
RIGVEDA = re.compile(r"\d+\.\d{3}\.\d{2}")


@dataclass(frozen=True)
class Unit:
    id: str  # the text's id and the verse's reference or block key: "vedic-chant-pavamana-suktam/9.001.01"
    hymn: int  # the Rigveda hymn (Pavamana Suktam); 0 for other texts
    verse: int
    label: str

    @property
    def key(self) -> str:
        return self.id.split("/", 1)[1]

    @property
    def block(self) -> str:
        """The block a unit is, or is a piece of: "4" for the piece "4.2" (a Rigveda reference is a whole block)."""
        piece = re.fullmatch(r"(.+)\.\d+", self.key)
        return piece.group(1) if piece and not RIGVEDA.fullmatch(self.key) else self.key


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


# The most letters one unit holds: about 12 seconds of chanting, as long as a Pavamana verse. Longer clips (20 s and
# more) were heard worse, the model sometimes repeating a phrase (2026-11-05)
MAX_LETTERS = 120


def pieces(lines: list[Line]) -> list[list[Line]]:
    """A long block's lines in runs of about even length, none much over MAX_LETTERS letters (a single longer line stays
    whole): one run for a short block, several for a long one (an anuvaka of the Rudram), never a long one and a
    scrap."""
    sizes = [len(plain(ln.orig)) + 1 for ln in lines]
    # (a block up to half as long again stays whole: an aarti's stanza, a Pavamana verse)
    if sum(sizes) <= MAX_LETTERS * 1.5:
        return [lines]
    count = -(-sum(sizes) // MAX_LETTERS)  # (rounded up)
    target = sum(sizes) / count
    runs: list[list[Line]] = [[]]
    filled = 0
    for ln, size in zip(lines, sizes, strict=True):
        # a new run once this one has its share, as long as runs are left to fill
        if runs[-1] and filled + size / 2 > target * len(runs) and len(runs) < count:
            runs.append([])
        runs[-1].append(ln)
        filled += size
    return runs


def keyed(text: Text) -> list[tuple[str, list[Line]]]:
    """A text's blocks with their keys: a Rigveda reference for Pavamana Suktam's verses (verses(), "" for its blocks
    without one), else a block's number ("132") or its kind and count ("once-7") (stanzas())."""
    out: list[tuple[str, list[Line]]] = []
    seen: dict[str, int] = {}
    referenced = bool(verses(text))
    for lines in blocks(text):
        if referenced:
            ref = next((m for m in (REF.search(ln.en) for ln in lines) if m), None)
            out.append((f"{ref.group(1)}.{ref.group(2)}.{ref.group(3)}" if ref else "", lines))
            continue
        name = lines[0].block
        number = re.fullmatch(r"(?:para )?(\d+)", name)
        if number:
            key = number.group(1)
        else:
            kind = name.split()[0]
            seen[kind] = seen.get(kind, 0) + 1
            key = f"{kind}-{seen[kind]}"
        out.append((key, lines))
    return out


def stanzas(text: Text) -> list[Unit]:
    """Every block of a text, in order, keyed by its number ("132") or its kind and count ("once-7"); a block too long
    for one clip in pieces of a few lines, keyed "4.1", "4.2", …"""
    out: list[Unit] = []
    for key, lines in keyed(text) if not verses(text) else []:
        number = re.fullmatch(r"\d+", key)
        runs = pieces(lines)
        for n, run in enumerate(runs, 1):
            label = " ".join(plain(ln.orig) for ln in run).strip()
            if label:
                piece = key if len(runs) == 1 else f"{key}.{n}"
                out.append(Unit(f"{text.id}/{piece}", 0, int(key) if number else 0, label))
    return out


def sung(text: Text) -> list[Unit]:
    """A text's units in the order they are chanted. With "refrain: every-verse" (an aarti) the dhrupad comes again
    after every stanza, and first of all with "refrain-before-first: yes", as the site shows it; otherwise each block
    once, in the file's order."""
    every = of(text)
    if text.header.get("refrain") != "every-verse":
        return every
    refrain = [u for u in every if u.block.startswith("dhrupad-")]
    out = (
        [u for u in every if u.block.startswith("once-") and every.index(u) < every.index(refrain[0])]
        if refrain
        else []
    )
    if text.header.get("refrain-before-first") == "yes":
        out += refrain
    for u in every:
        if u.block.isdigit():
            # (a long stanza in pieces: the dhrupad after its last)
            last = u is every[-1] or every[every.index(u) + 1].block != u.block
            out += [u, *refrain] if last else [u]
    return out


def of(text: Text) -> list[Unit]:
    """A text's units: its referenced verses if it has them (Pavamana Suktam), else its blocks."""
    return verses(text) or stanzas(text)


def select(every: list[Unit], parts: str) -> list[Unit]:
    """The units a recording chants, in its order: keys and ranges separated by spaces,
    "132..183 once-7 once-2..once-5". A range runs in the text's order, both ends included. A block in pieces is named
    whole ("4", all its pieces) or by piece ("4.2")."""
    first_of: dict[str, int] = {}
    last_of: dict[str, int] = {}
    for i, u in enumerate(every):
        for k in {u.key, u.block}:
            first_of.setdefault(k, i)
            last_of[k] = i
    out: list[Unit] = []
    for part in parts.split():
        first, _, last = part.partition("..")
        for key in (first, last or first):
            if key not in first_of:
                raise KeyError(f"no block {key!r} in {every[0].id.split('/')[0] if every else 'the text'}")
        a, b = first_of[first], last_of[last or first]
        if b < a:
            raise ValueError(f"{part}: the range runs backwards")
        out += every[a : b + 1]
    return out


def chanted(texts: dict[str, Text], ids: tuple[str, ...], parts: tuple[str, ...]) -> list[Unit]:
    """A recording's units in the order chanted: for each of its texts in turn, the blocks it names (select()), or
    the whole text as it is chanted (sung())."""
    return [u for t, p in zip(ids, parts, strict=True) for u in (select(of(texts[t]), p) if p else sung(texts[t]))]


def candidates(texts: dict[str, Text], ids: tuple[str, ...]) -> list[Unit]:
    """Every unit of a recording's texts, each once: what a clip's verse is looked for among."""
    return [u for t in dict.fromkeys(ids) for u in of(texts[t])]
