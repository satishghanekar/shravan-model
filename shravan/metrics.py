"""How good a model's hearing is, on clips whose words are known.

- Letters wrong (character error rate): edits needed to turn what the model wrote into the label, over the label's
  length, across all clips together. What it wrote is cleaned by the same rules as the labels first, so a stray danda
  or svara mark is not counted against it.
- Verse found: whether, of every verse of the text, the one closest to what the model wrote (fewest letters wrong) is
  the right one. It stands for what the app needs: to know where the chanting is.
"""

from __future__ import annotations

import jiwer

from shravan.labels import plain


def clean(hypothesis: str) -> str:
    return plain(hypothesis.replace(",", " ").replace(".", " "))


def letters_wrong(labels: list[str], hypotheses: list[str]) -> float:
    return float(jiwer.cer(labels, [clean(h) or "-" for h in hypotheses]))


def closest(hypothesis: str, candidates: list[str]) -> int:
    """The index of the candidate label nearest to what the model wrote."""
    h = clean(hypothesis) or "-"
    return min(range(len(candidates)), key=lambda i: jiwer.cer(candidates[i], h))


def verses_found(truth: list[int], hypotheses: list[str], candidates: list[str]) -> float:
    found = sum(closest(h, candidates) == t for h, t in zip(hypotheses, truth, strict=True))
    return found / max(len(truth), 1)


def validation_score(published: float, family: float | None) -> float:
    """What the best checkpoint is chosen on: the published and the family clips counted alike, however many of each."""
    return published if family is None else (published + family) / 2
