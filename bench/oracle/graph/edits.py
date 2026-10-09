"""Edits of an oracle answer (#2951 graph oracle): dropping one parent of a reader in its graph (an edge), the move RL's
credit and refinement measure.

A graph answer writes subcomponents as "<p:L.S.I>" tokens; the ones in a reader's parents (strings or lists, not the
(position, reader) keys) are its edges. An edit drops occurrences of them from the program text (and tidies the lists
they stood in). credit() compares a sample of single drops with the answer; refine() repeats rounds of credit, keeping
the drops that improve it. Both rank scores by a key (score.order at the question's eps: lower is better).
"""

from __future__ import annotations

import random
import re

NAME = re.compile(r"<p:\d+\.(?:q|k|v|o|fc|down)\.(?:\d+|rest)>")
EMPTY_ITEM = re.compile(r'(?<=[\[,])(\s*)(?:""|\'\')(?=\s*[,\]])')  # a list item left empty by a drop


KEY = re.compile(r"\(\s*[^,()]+?\s*,\s*[\"'](<p:[^>]+>)[\"']\s*\)")  # a (position, reader) key


def names(source: str) -> list[tuple[int, int]]:
    """The spans [start, end) of the subcomponents in `source` that are parents (outside every (position, reader)
    key)."""
    keys = [m.span(1) for m in KEY.finditer(source)]
    return [m.span() for m in NAME.finditer(source) if m.span() not in keys]


def drop(source: str, spans) -> str:
    """`source` without the names at `spans`, its lists tidied (no empty items, no leading or trailing commas)."""
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(source[last:a])
        last = b
    out.append(source[last:])
    text = EMPTY_ITEM.sub(r"\1", "".join(out))
    for pattern, repl in ((r",(\s*),", r",\1"), (r"\[(\s*),\s*", r"[\1"), (r",(\s*)\]", lambda m: (m[1] if "\n" in m[1] else "") + "]")):
        while re.search(pattern, text):
            text = re.sub(pattern, repl, text)
    return text


def credit(source: str, score, key, k: int = 0, rng: random.Random | None = None) -> tuple[tuple, dict[tuple[int, int], int]]:
    """(the answer's key, {name span: +1 when dropping the name makes the answer worse, -1 when better, 0 when neither})
    for every name or for k sampled (score: a list of sources -> scores; key: a score -> its rank, lower better)."""
    spans = names(source)
    if k and len(spans) > k:
        spans = (rng or random.Random(0)).sample(spans, k)
    keys = [key(r) for r in score([source] + [drop(source, [s]) for s in spans])]
    return keys[0], {s: (keys[i + 1] > keys[0]) - (keys[i + 1] < keys[0]) for i, s in enumerate(spans)}


def refine(source: str, score, key, rounds: int, k: int = 0, rng: random.Random | None = None) -> tuple[str, tuple, int]:
    """(the refined answer, its key, the names dropped): each round credits the answer's names (k sampled) and drops
    every one whose drop improves it, or the first of them alone when dropping them together does not."""
    best = source
    base = key(score([source])[0])
    dropped = 0
    for _ in range(rounds):
        _, signs = credit(best, score, key, k, rng)
        gains = [s for s, v in sorted(signs.items()) if v < 0]
        if not gains:
            break
        for trial in (gains, gains[:1]):
            text = drop(best, trial)
            t = key(score([text])[0])
            if t < base:
                best, base, dropped = text, t, dropped + len(trial)
                break
        else:
            break
    return best, base, dropped
