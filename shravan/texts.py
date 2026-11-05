"""The chant texts, read from the copy in texts/ (made by scripts/sync_texts.py from the Nitya Upasana site).

The format is the site's (its README, "How to add a text"): a header of "name: value" lines, a "---" line, then blocks
named in brackets ([1], [para 2], [dhrupad], [once], [note: … | …]), each holding pairs of lines, the original then its
everyday English. This module only reads them; it never changes a text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# A line in an Indian script, as the site tells an original line from its English one
INDIC = re.compile(r"[ऀ-෿]")


@dataclass(frozen=True)
class Line:
    """One line pair: its block's name and number within the text, the original line and its everyday English."""

    block: str
    index: int
    orig: str
    en: str


@dataclass
class Text:
    id: str
    title: str
    title_en: str
    language: str
    header: dict[str, str]
    lines: list[Line] = field(default_factory=list)


class TextError(ValueError):
    pass


def parse(raw: str, name: str = "text") -> Text:
    """A text file's contents → Text. Raises TextError, naming the line, on anything the site would refuse."""
    rows = raw.split("\n")
    header: dict[str, str] = {}
    i = 0
    while i < len(rows) and rows[i] != "---":
        row = rows[i]
        if row:
            m = re.fullmatch(r"([a-z-]+): (.*)", row)
            if not m:
                raise TextError(f'{name} line {i + 1}: expected "name: value" in the header, or "---" to end it')
            header[m.group(1)] = m.group(2)
        i += 1
    if i == len(rows):
        raise TextError(f'{name}: no "---" line after the header')
    for key in ("id", "title", "title-en", "language"):
        if key not in header:
            raise TextError(f'{name}: header field "{key}" is missing')

    text = Text(header["id"], header["title"], header["title-en"], header["language"], header)
    block = ""
    orig: str | None = None
    index = 0
    for n, row in enumerate(rows[i + 1 :], start=i + 2):
        if row == "":
            if orig is not None:
                raise TextError(f"{name} line {n}: an original line with no English line after it")
            continue
        head = re.fullmatch(r"\[(.*)\]", row)
        if head:
            if head.group(1) != "refrain":  # [refrain] marks lines inside a block; it starts no new block
                block, index = head.group(1), 0
            continue
        if not block:
            raise TextError(f"{name} line {n}: a line outside a block")
        if orig is None:
            if not INDIC.search(row):
                raise TextError(f"{name} line {n}: expected an original line (in an Indian script) here")
            orig = row
        else:
            if INDIC.search(row):
                raise TextError(f"{name} line {n}: expected the everyday English line here")
            index += 1
            text.lines.append(Line(block, index, orig, row))
            orig = None
    if orig is not None:
        raise TextError(f"{name}: the last original line has no English line after it")
    return text


def load_all(folder: Path) -> dict[str, Text]:
    """Every text in a folder, by id."""
    texts = {}
    for path in sorted(folder.glob("*.txt")):
        text = parse(path.read_text(encoding="utf-8"), path.name)
        texts[text.id] = text
    return texts
