"""Where the chanting is, worked out exactly as Listen does it on the phone, so scripts/follow.py scores the app's own
method (the site's js/core/align.js and js/core/follow.js, line for line; tests/fixtures/listen.json holds what the
site's code answers, and tests/test_follow.py checks this module gives the same).

Listen hears the last few seconds every half second. When it knows roughly where the chanting is (the mark), it lines
the model's scores up with the text near there (align(): every letter of the text scored against every 20 ms step,
CTC's forced alignment, free to start and end anywhere in the stretch). When that doesn't fit, or it doesn't know
yet, it reads the letters the model would write and looks for them in the texts, allowing for mistakes (locate()).

A text here is a Doc: the said letters of its units end to end, with no spaces (the model and the book split words
differently where they join), each unit's end, and which units are the dhrupad (said again after every stanza, so they
never tell where the chanting is: the mark waits).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shravan.texts import Line, Text
from shravan.units import Unit, keyed

# the same three patterns as the site's js/core/follow.js letters()
VEDIC_MARKS = re.compile(r"[॑-॔᳐-᳿]")
VERSE_NUMBER = re.compile(r"[॥।|]+\s*[०-९0-9]+(?:\.[०-९0-9]+)*\s*(?:[॥।|]+)?")
NOT_SAID = re.compile(r"[^ऀ-ॣॱ-ॿ]|ऽ")


def letters(line: str) -> str:
    """The letters that are said, with nothing between them: a line of a text, or what the model heard."""
    t = re.sub(r"<[^>]*>", " ", line)
    t = re.sub(r"&[#a-z0-9]+;", " ", t, flags=re.I)
    t = unicodedata.normalize("NFC", t)
    t = VERSE_NUMBER.sub(" ", t)
    t = VEDIC_MARKS.sub("", t)
    return NOT_SAID.sub("", t)


@dataclass
class Doc:
    """A text to search: `text` its units' letters end to end, `ends[i]` where unit i's letters end, `refrain[i]`
    whether it is the dhrupad."""

    text: str = ""
    ends: list[int] = field(default_factory=list)
    refrain: list[bool] = field(default_factory=list)


def make_doc(lines: list[tuple[str, bool]]) -> Doc:
    """A Doc from (line, is it the dhrupad) pairs, in the order the page shows them."""
    doc = Doc()
    for line, refrain in lines:
        doc.text += letters(line)
        doc.ends.append(len(doc.text))
        doc.refrain.append(refrain)
    return doc


def line_at(doc: Doc, at: int) -> int:
    """The unit a letter is in."""
    lo, hi = 0, len(doc.ends) - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if doc.ends[mid] > at:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ==== Lining the sound up with the text (js/core/align.js) ====


def log_probs(scores: np.ndarray) -> np.ndarray:
    """Each step's scores (steps × outputs) as log-probabilities, so they can be added along a path."""
    s: np.ndarray = scores.astype(np.float64)
    top = s.max(axis=1, keepdims=True)
    out: np.ndarray = (s - (top + np.log(np.exp(s - top).sum(axis=1, keepdims=True)))).astype(np.float32)
    return out


def target(text: str, ids: dict[str, int], start: int) -> tuple[list[int], list[int]]:
    """A stretch of text as the model's outputs, and where each is in the text (a letter it has no output for is left
    out)."""
    out, at = [], []
    for i, ch in enumerate(text):
        if ch in ids:
            out.append(ids[ch])
            at.append(start + i)
    return out, at


@dataclass(frozen=True)
class Aligned:
    end: int  # the letter the window ends on, an index into the stretch's ids
    step: int  # the step that letter began on
    letters: int  # how many letters the window covered
    fit: float  # log-probability a step, less the model's own likeliest: 0 is as good as its own reading


def align(lp: np.ndarray, ids: list[int], blank: int, word: int) -> Aligned | None:
    """The best path of the window (log-probabilities, steps × outputs) through a stretch of text: the window may start
    and end anywhere in it. The word break, said or not, counts as the blank. None when the stretch is empty."""
    steps, n = lp.shape[0], len(ids)
    if not n or not steps:
        return None
    lp64: np.ndarray = lp.astype(np.float64)
    size = 2 * n + 1
    states = np.arange(size)
    odd = states % 2 == 1
    letter = np.where(odd, (states - 1) // 2, -1)
    idx = np.array(ids)
    # states 2k: the blank before letter k; 2k + 1: letter k
    emit_ids = np.where(odd, idx[np.clip(letter, 0, None)], 0)
    # a letter may skip the blank before it when the letter before it is another one
    skip: np.ndarray = np.zeros(size, dtype=bool)
    skip[3::2] = idx[1:] != idx[:-1]

    def emit(t: int) -> np.ndarray:
        row = lp64[t]
        out: np.ndarray = np.where(odd, row[emit_ids], max(row[blank], row[word]))
        return out

    cur = emit(0)
    last_letter = np.where(odd, letter, -1)
    last_step = np.where(odd, 0, -1)
    first = np.where(odd, letter, states // 2)
    for t in range(1, steps):
        # stay, or come from the state before, or (a letter) skip the blank: the later only when strictly better,
        # as the site breaks ties
        v = cur.copy()
        frm = states.copy()
        one = np.full(size, -np.inf)
        one[1:] = cur[:-1]
        better = one > v
        v = np.where(better, one, v)
        frm = np.where(better, states - 1, frm)
        two = np.full(size, -np.inf)
        two[2:] = cur[:-2]
        better = skip & (two > v)
        v = np.where(better, two, v)
        frm = np.where(better, states - 2, frm)
        entered = odd & (frm != states)
        last_letter = np.where(entered, letter, last_letter[frm])
        last_step = np.where(entered, t, last_step[frm])
        first = first[frm]
        cur = v + emit(t)
    best = int(np.argmax(cur))
    own = float(sum(float(r.max()) for r in lp64))
    if last_letter[best] < 0:
        return None
    return Aligned(
        int(last_letter[best]),
        int(last_step[best]),
        int(last_letter[best] - first[best] + 1),
        (float(cur[best]) - own) / steps,
    )


def latest(lp: np.ndarray, steps: int) -> np.ndarray:
    """The last `steps` steps of a window, to line up on their own."""
    return lp[-steps:] if lp.shape[0] > steps else lp


# ==== Looking for the letters heard (js/core/follow.js) ====


def match(heard: str, text: str, start: int = 0, stop: int | None = None) -> tuple[int, int]:
    """Where the stretch of text[start:stop] closest to `heard` ends (just after its last letter), and how many letters
    it differs by: the edit distance free to start and end anywhere (Sellers' algorithm), on the earliest best end."""
    a = max(0, start)
    b = min(len(text), len(text) if stop is None else stop)
    n = b - a
    seg = np.array([ord(c) for c in text[a:b]], dtype=np.int64)
    prev: np.ndarray = np.zeros(n + 1, dtype=np.int64)
    ramp = np.arange(n + 1)
    for i, ch in enumerate(heard, 1):
        diag = prev[:-1] + (seg != ord(ch))
        up: np.ndarray = prev[1:] + 1
        cand: np.ndarray = np.empty(n + 1, dtype=np.int64)
        cand[0] = i
        cand[1:] = np.minimum(diag, up)
        # a step left costs one more each time: the running least of (cand - place), plus the place
        prev = np.minimum.accumulate(cand - ramp) + ramp
    best, end = len(heard), a
    if n:
        k = int(np.argmin(prev[1:])) + 1
        if prev[k] < best:
            best, end = int(prev[k]), a + k
    return end, best


@dataclass(frozen=True)
class Settings:
    """Listen's own numbers (js/features/listen.js and js/core/follow.js), from config.yaml's listen: section."""

    window: float = 4.0
    back: int = 60
    forth: int = 250
    fit: float = -0.6
    cover: int = 10
    recent: float = 1.0
    tail: float = 2.5
    least: int = 10
    ahead: int = 400
    near: float = 0.4
    text: float = 0.35
    any: float = 0.3
    pace: float = 7.0
    step_ms: float = 20.0

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Settings:
        return cls(**{k: type(getattr(cls, k))(v) for k, v in raw.items()})


@dataclass(frozen=True)
class Found:
    doc: int
    end: int  # the letter just after what was heard, in docs[doc].text; -1 when held
    held: bool  # the dhrupad heard: the mark waits where it is
    how: str  # "aligned", or locate()'s scope: "near", "text", "any"


def locate(heard: str, docs: list[Doc], here: tuple[int, int] | None, s: Settings) -> Found | None:
    """Where the letters heard are: near the place last found (doc, unit) first, then in its whole text, then in any
    text, each a little stricter."""
    h = letters(heard)
    if len(h) < s.least:
        return None

    def look(k: int, start: int, stop: int, most: float, scope: str) -> tuple[float, Found] | None:
        doc = docs[k]
        end, cost = match(h, doc.text, start, stop)
        rate = cost / len(h)
        if rate > most or end <= start:
            return None
        if doc.refrain[line_at(doc, end - 1)] and here:
            return rate, Found(here[0], -1, True, scope)
        return rate, Found(k, end, False, scope)

    if here and here[0] < len(docs):
        d = docs[here[0]]
        # (the site reads ends[line - 2] as 0 when there is none, and ends[line] + AHEAD past the end as the end)
        start = d.ends[here[1] - 2] if here[1] >= 2 else 0
        near = look(here[0], start, d.ends[here[1]] + s.ahead, s.near, "near")
        if near:
            return near[1]
        whole = look(here[0], 0, len(d.text), s.text, "text")
        if whole:
            return whole[1]
    best: tuple[float, Found] | None = None
    for k, doc in enumerate(docs):
        found = look(k, 0, len(doc.text), s.any, "any")
        if found and (best is None or found[0] < best[0]):
            best = found
    return best[1] if best else None


# ==== One hearing, as Listen's loop takes it ====


@dataclass(frozen=True)
class Model:
    """What the place-finding needs from the model: each letter's output, the blank and the word break."""

    ids: dict[str, int]
    names: list[str]  # each output's letter, "" for one that is not a letter
    blank: int
    word: int


def read(choices: list[int], m: Model) -> str:
    """The model's likeliest outputs as words, as the site's readLetters() reads them."""
    out, before = "", -1
    for i in choices:
        if i != before and i != m.blank:
            if i == m.word:
                out += " "
            elif m.names[i]:
                out += m.names[i]
        before = i
    return " ".join(out.split())


def hear(scores: np.ndarray, docs: list[Doc], mark: tuple[int, float] | None, m: Model, s: Settings) -> Found | None:
    """Where one window's scores (steps × outputs) put the chanting, given the mark (doc, the letter it has reached now)
    or None when lost: the window lined up near the mark; if that doesn't fit, the letters of its last `tail` seconds
    looked for, then all its letters (listen.js loop() and alignAt())."""
    lp = log_probs(scores)
    if mark is not None:
        doc = docs[mark[0]]
        here = mark[1]
        start = max(0, int(np.floor(here - s.back)))
        ids, at = target(doc.text[start : int(np.ceil(here + s.forth))], m.ids, start)
        a = align(lp, ids, m.blank, m.word)
        if a and a.fit >= s.fit and a.letters >= s.cover:
            fresh = align(latest(lp, round(s.recent * 1000 / s.step_ms)), ids, m.blank, m.word)
            if fresh and fresh.fit >= s.fit:
                pos = at[a.end] + 1
                # the dhrupad said again (lined up back where it is written once): the mark waits
                if doc.refrain[line_at(doc, pos - 1)] and pos < here - s.cover:
                    return Found(mark[0], -1, True, "aligned")
                return Found(mark[0], pos, False, "aligned")
    choices = [int(i) for i in scores.argmax(axis=1)]
    here_unit = (mark[0], line_at(docs[mark[0]], max(0, int(mark[1]) - 1))) if mark is not None else None
    cut = max(0, len(choices) - round(s.tail * 1000 / s.step_ms))
    if mark is not None and cut:
        tail = read(choices[cut:], m)
        if len(letters(tail)) >= s.least:
            found = locate(tail, docs, here_unit, s)
            if found:
                return found
    return locate(read(choices, m), docs, here_unit, s)


# ==== A recording's texts as Listen sees them, and whether a place found is right ====


@dataclass
class Book:
    """The texts a recording chants, each a Doc in the order the site shows its lines (js/text/content.js): with
    "refrain: every-verse", what comes before the dhrupad, the dhrupad, then each stanza with the dhrupad again after
    it. `keys[d][i]` is line i's block key (units.keyed()), so a place found can be checked against the unit chanted."""

    ids: list[str]
    docs: list[Doc]
    keys: list[list[str]]
    every_verse: list[bool]

    def doc_of(self, unit_id: str) -> int:
        return self.ids.index(unit_id.split("/", 1)[0])

    def span(self, unit_id: str) -> tuple[int, int] | None:
        """Where a unit's block is first shown in its text, in letters (start, end); None if it isn't."""
        d = self.doc_of(unit_id)
        key = Unit(unit_id, 0, 0, "").block
        lines = [i for i, k in enumerate(self.keys[d]) if k == key]
        if not lines:
            return None
        ends = self.docs[d].ends
        return (ends[lines[0] - 1] if lines[0] else 0), ends[lines[0]]

    def is_refrain(self, unit_id: str) -> bool:
        """Whether a unit is the dhrupad of an aarti (said after every stanza)."""
        d = self.doc_of(unit_id)
        key = Unit(unit_id, 0, 0, "").block
        return self.every_verse[d] and any(
            r for k, r in zip(self.keys[d], self.docs[d].refrain, strict=True) if k == key
        )


def site_order(text: Text) -> list[tuple[str, Line, bool]]:
    """A text's lines as the site shows them: (block key, line, is it the dhrupad)."""
    blocks = keyed(text)
    if text.header.get("refrain") != "every-verse":
        return [(k, ln, ln.refrain) for k, lines in blocks for ln in lines]
    kind = [lines[0].block.split()[0] for _, lines in blocks]
    d = next(i for i, k in enumerate(kind) if k not in ("once", "note"))
    dh = blocks[d]
    rep = blocks[d + 1] if d + 1 < len(blocks) and blocks[d + 1][1][0].block == "dhrupad repeated" else dh
    rest = blocks[d + (2 if rep is not dh else 1) :]

    def again() -> list[tuple[str, Line, bool]]:
        return [(dh[0], ln, True) for ln in rep[1]]

    out = [(k, ln, ln.refrain) for k, lines in blocks[:d] for ln in lines]
    out += [(dh[0], ln, True) for ln in dh[1]]
    if text.header.get("refrain-before-first") == "yes":
        out += again()
    for (key, lines), k in zip(rest, kind[d + (2 if rep is not dh else 1) :], strict=True):
        out += [(key, ln, ln.refrain) for ln in lines]
        if k not in ("once", "dhrupad", "note"):
            out += again()
    return out


def book(texts: list[Text]) -> Book:
    docs, keys, every = [], [], []
    for t in texts:
        lines = site_order(t)
        docs.append(make_doc([(ln.orig, refrain) for _, ln, refrain in lines]))
        keys.append([k for k, _, _ in lines])
        every.append(t.header.get("refrain") == "every-verse")
    return Book([t.id for t in texts], docs, keys, every)


def right(b: Book, unit_id: str, found: Found | None) -> bool:
    """Whether a place found is the unit being chanted: the same block of the same text, or, while the dhrupad is said
    again, the mark held or the dhrupad found."""
    if found is None:
        return False
    if b.is_refrain(unit_id):
        return found.held or (
            found.doc == b.doc_of(unit_id) and b.docs[found.doc].refrain[line_at(b.docs[found.doc], found.end - 1)]
        )
    if found.held or found.doc != b.doc_of(unit_id):
        return False
    return b.keys[found.doc][line_at(b.docs[found.doc], found.end - 1)] == Unit(unit_id, 0, 0, "").block


def mark_at(b: Book, rows: list[dict[str, Any]], i: int, t: float, s: Settings) -> tuple[int, float] | None:
    """Where Listen's mark is at `t` seconds while row i (the units as aligned in the recording, in order) is being
    chanted, when it followed the chanting until then: at the end of the last unit before that is not the dhrupad
    said again, moved on at Listen's starting pace since that unit ended, or waiting while the dhrupad is said. None
    for the first unit (nothing followed yet: the place must be found)."""
    j = next((k for k in range(i - 1, -1, -1) if not b.is_refrain(rows[k]["id"])), None)
    if j is None:
        return None
    where = b.span(rows[j]["id"])
    if where is None:
        return None
    d = b.doc_of(rows[j]["id"])
    waiting = b.is_refrain(rows[i]["id"]) or any(b.is_refrain(rows[k]["id"]) for k in range(j + 1, i))
    moved = 0.0 if waiting else s.pace * max(0.0, t - float(rows[j]["end"]))
    return d, min(float(len(b.docs[d].text)), where[1] + moved)
