"""The English reader (#2951 graph oracle): how much an answer's plain English tells about the model.

A frozen copy of the base oracle model (never trained) reads the text and only the English of an answer (its docstring
and its comment lines, numbered as steps). For each of the verifier's events (score.py's "events", native.events) it
gives its probability that the model's probability of its most likely next token goes down: when one token is replaced,
and when one token is replaced while one of the answer's steps is held at its values on the original text. The English
scores the bits it saves: the reader's importance-weighted mean log loss, in bits, on what the model actually does
without the English minus with it. English that says how the prediction depends on the text, and what each step
carries, saves bits; vague English saves none; wrong English costs. No judge grades style, and the answer never chooses
the questions.

  Reader(generate) with generate(prompts: list[str]) -> per prompt {token string: log-probability} of the next token.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ASK = ("A language model reads this text (its tokens, by position) and predicts the next token after the last one; its "
       "most likely next token is {top}.\ntokens = {tokens}\n\n{english}"
       "Question: if the token at position {position}, {old}, were replaced by {new}{hold}, would the probability the model "
       "gives {top} go down? Answer Yes or No.")
HOLD = ", while the subcomponents of step {k} of the explanation kept their values on the original text"


def english(score: dict) -> str:
    """An answer's English: its docstring, then its comment lines numbered as steps."""
    notes = [n for n in score.get("notes") or [] if n]
    parts = [score.get("explanation") or ""] + [f"Step {k}: {n}" for k, n in enumerate(notes, 1)]
    return "\n".join(p for p in parts if p.strip())


class Reader:
    def __init__(self, generate, tokens_of):
        """generate: list of prompt strings -> per prompt {token string: log-probability} of the next token (the
        frozen base model, top candidates); tokens_of: token ids -> token strings of the target model."""
        self.generate, self.tokens_of = generate, tokens_of

    def _p_yes(self, prompts: list[str]) -> list[float]:
        out = []
        for lp in self.generate(prompts):
            yes = sum(math.exp(v) for k, v in lp.items() if k.strip().lower() == "yes")
            no = sum(math.exp(v) for k, v in lp.items() if k.strip().lower() == "no")
            out.append(yes / (yes + no) if yes + no > 0 else 0.5)
        return out

    def bits(self, task: dict, texts: list[str], events) -> list[float | None]:
        """Per English text, the bits it saves the reader on the events (events: one list for every text, or
        events[j] text j's own; None for a text without events)."""
        per = events if events and isinstance(events[0], list) else [events] * len(texts)
        prompt = task["prompts"][0]
        strings = self.tokens_of(prompt["token_ids"])
        top = repr(prompt["model_top"][0][0][0])

        def ask(eng: str, e: dict) -> str:
            head = f"An explanation of how the model computes this prediction:\n{eng}\n\n" if eng else ""
            return ASK.format(top=top, tokens=repr(strings), english=head, position=e["position"], old=repr(strings[e["position"]]),
                              new=repr(self.tokens_of([e["new"]])[0]), hold=HOLD.format(k=e["hold"]) if e.get("hold") else "")

        def loss(ps: list[float], evs: list[dict]) -> float:  # the events' mean, by their importance weights
            return sum(e.get("weight", 1.0) * -math.log2(max(p if e["down"] else 1 - p, 1e-12)) for p, e in zip(ps, evs)) / sum(e.get("weight", 1.0) for e in evs)

        flat, spans = [], []
        for t, evs in zip(texts, per):
            spans.append((len(flat), len(evs)))
            flat += [ask("", e) for e in evs] + [ask(t, e) for e in evs]
        ps = self._p_yes(flat) if flat else []
        out = []
        for (s0, n), evs in zip(spans, per):
            out.append(None if not n else loss(ps[s0:s0 + n], evs) - loss(ps[s0 + n:s0 + 2 * n], evs))
        return out
