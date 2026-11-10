"""The CTC model's letters: a wav2vec2 model that writes one letter (or nothing) for every 20 ms of sound.

It hears only the sound it is given and says, for each 20 ms, which letter is being said: quick on a phone, and the
letters come with their moments, which is what following along needs (2026-11-06).

Its alphabet is the pretrained model's (Vakyansh's Sanskrit model writes Devanagari letter by letter, "|" between
words) with any letter our labels use that it lacks added at the end; what a label holds beyond the letters (commas,
full stops, brackets) is left out, and a rare Vedic sign is written as the everyday one.
"""

from __future__ import annotations

from collections.abc import Iterable

WORD = "|"
# the tokens that are not letters, in the pretrained model's vocabulary
SPECIAL = ("<s>", "</s>", "<unk>", "<pad>")
# letters written as another: the Vedic anusvara (ꣳ) is said as the plain one
SAME = {"ꣳ": "ं"}


def text(label: str) -> str:
    """A label as the CTC model writes it: Devanagari letters, words joined by "|", nothing else."""
    out = []
    for ch in label:
        ch = SAME.get(ch, ch)
        if ch == " ":
            out.append(WORD)
        elif "ऀ" <= ch <= "ॿ" and ch not in "।॥":
            out.append(ch)
    return WORD.join(w for w in "".join(out).split(WORD) if w)


def alphabet(vocab: dict[str, int], labels: Iterable[str]) -> dict[str, int]:
    """The pretrained vocabulary, with every letter of the labels it lacks added after it, in order."""
    out = dict(vocab)
    for ch in sorted({ch for label in labels for ch in text(label)} - set(vocab)):
        out[ch] = len(out)
    return out


def encode(label: str, alpha: dict[str, int]) -> list[int]:
    return [alpha[ch] for ch in text(label)]


def decode(ids: Iterable[int], alpha: dict[str, int], blank: int) -> str:
    """The model's choice for each 20 ms → words: a letter held over several steps counts once, the blank (nothing
    said) separates two of the same letter, "|" is a space."""
    letters = {i: ch for ch, i in alpha.items()}
    out, last = [], -1
    for i in ids:
        if i != last and i != blank:
            ch = letters.get(i, "")
            if ch not in SPECIAL:
                out.append(" " if ch == WORD else ch)
        last = i
    return " ".join("".join(out).split())


def blank_of(chosen: Iterable[Iterable[int]]) -> int:
    """The model's "nothing said" token: the one it writes most on real chanting (`chosen`, its choices for each clip).
    Its config can't be trusted for it: Vakyansh's models, converted from fairseq, name <pad> but write <s> (id 0), and
    training against the wrong one taught the model that every pause was a mistake, so within a few dozen steps it
    wrote nothing at all (2026-11-06)."""
    counts: dict[int, int] = {}
    for clip in chosen:
        for i in clip:
            counts[i] = counts.get(i, 0) + 1
    return max(counts, key=lambda i: counts[i])
