"""Where each verse starts and ends in a long recording, found by forced alignment against the known text.

The aligner is Meta's MMS model in torchaudio (MMS_FA): it gives, for every 20 ms of audio, how likely each romanized
letter is, and forced alignment finds the best path through the known letters. A whole recording at once would need
tens of gigabytes, so it is aligned in windows: a window takes the next verses and a stretch of audio somewhat longer
than they should take, with a wildcard (*) at each end to absorb pauses, music or audio that belongs elsewhere. The
first verses of the window are kept, and the next window starts where the last kept one ended.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio.functional as F

from shravan.units import Unit

STAR = "*"


@dataclass(frozen=True)
class Placed:
    unit: Unit
    start: float  # seconds
    end: float
    score: float  # mean probability of the unit's letters, 0–1: low means a doubtful alignment


def romanize_words(label: str, romanize: object, dictionary: dict[str, int]) -> list[str]:
    """A plain Devanagari label → the aligner's romanized words, keeping only letters it knows (not its blank, "-",
    which the romanizer can write)."""
    roman = romanize.romanize_string(label).lower()  # type: ignore[attr-defined]
    words = ("".join(c for c in w if dictionary.get(c, 0)) for w in roman.split())
    return [w for w in words if w]


def emissions(
    model: torch.nn.Module, wave: torch.Tensor, device: str, chunk_s: float = 30.0, rate: int = 16000
) -> tuple[torch.Tensor, torch.Tensor]:
    """The aligner's letter probabilities for a whole recording, 30 s at a time; and the time (s) each frame starts."""
    step = int(chunk_s * rate)
    parts, times = [], []
    for at in range(0, wave.shape[-1], step):
        piece = wave[..., at : at + step]
        if piece.shape[-1] < rate // 10:  # a last sliver too short to hear
            break
        with torch.inference_mode():
            em, _ = model(piece.to(device))
        em = em[0].cpu()
        parts.append(em)
        seconds = piece.shape[-1] / rate
        times.append(at / rate + torch.arange(em.shape[0]) * (seconds / em.shape[0]))
    return torch.cat(parts), torch.cat(times)


def align_window(emission: torch.Tensor, units: list[list[int]]) -> list[tuple[int, int, float]]:
    """Force-align some units' tokens (each a list of token ids, words already joined) inside one stretch of emission,
    with a wildcard at both ends; return each unit's first frame, last frame and mean letter probability."""
    star = emission.shape[-1] - 1  # MMS_FA's wildcard is its last token
    targets = [star] + [t for u in units for t in u] + [star]
    path, scores = F.forced_align(emission[None], torch.tensor([targets], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp())
    out = []
    i = 1  # spans[0] is the opening wildcard
    for u in units:
        mine = spans[i : i + len(u)]
        out.append((mine[0].start, mine[-1].end, float(sum(s.score for s in mine) / len(mine))))
        i += len(u)
    return out


def align(
    emission: torch.Tensor,
    times: torch.Tensor,
    units: list[Unit],
    tokens: list[list[int]],
    window: int = 20,
    keep: int = 15,
) -> list[Placed]:
    """Every unit placed in the recording, window by window."""
    letters = sum(len(t) for t in tokens)
    frames_per_letter = emission.shape[0] / max(letters, 1)
    placed: list[Placed] = []
    at, i = 0, 0
    while i < len(units):
        take = tokens[i : i + window]
        last = i + len(take) >= len(units)
        expected = sum(len(t) for t in take) * frames_per_letter
        end = emission.shape[0] if last else min(emission.shape[0], at + int(expected * 1.6) + 1500)
        found = align_window(emission[at:end], take)
        keep_n = len(take) if last else keep
        for k in range(keep_n):
            s, e, score = found[k]
            placed.append(Placed(units[i + k], float(times[at + s]), float(times[min(at + e, len(times) - 1)]), score))
        at += found[keep_n - 1][1]
        i += keep_n
    return placed
