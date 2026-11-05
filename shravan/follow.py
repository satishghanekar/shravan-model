"""Where in a text the chanting is, from a few seconds of what the model heard: what the app does to follow along.

The text's units are laid end to end; what was heard is matched against them, letters and all, allowing for mistakes
(a fuzzy search for the best matching stretch, rapidfuzz). The chanting is at the end of that stretch: the unit it ends
in. Searching the whole text is what the app does when it has lost its place; following, it searches only a few units
around the last one it found.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from shravan.metrics import clean


def where(heard: str, labels: list[str]) -> int:
    """The index of the label in which the best match of what was heard ends; -1 if nothing was heard."""
    h = clean(heard)
    if not h:
        return -1
    text, ends = "", []
    for label in labels:
        text += label + " "
        ends.append(len(text))
    found = fuzz.partial_ratio_alignment(h, text)
    if found is None:
        return -1
    end = found.dest_end - 1
    return next(i for i, e in enumerate(ends) if end < e)


def nearby(last: int, count: int, before: int = 1, after: int = 3) -> range:
    """The units the app searches while following: from one before the last one found to a few after it."""
    return range(max(last - before, 0), min(last + after + 1, count))
