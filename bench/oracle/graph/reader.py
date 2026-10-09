"""The English reader (#2951 graph oracle): how much an answer's plain English tells about the model.

A frozen copy of the base oracle model (never trained) reads the text and only the English of an answer (its docstring
and the comment lines of its steps). For each changed prompt of the verifier that replaces a token (score.py's
"events"), it gives its probability that the model's most likely next token changes. The English scores the bits it
saves: the reader's mean log loss, in bits, on what the model actually does (the flips) without the English minus with
it. English that says how the prediction depends on the text saves bits; vague English saves none; wrong English costs.
No judge grades style, and the answer never chooses the questions.

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
       "Question: if the token at position {position}, {old}, were replaced by {new}, would the model's most likely next "
       "token change? Answer Yes or No.")


def english(score: dict) -> str:
    """An answer's English: its docstring, then its step comments."""
    parts = [score.get("explanation") or ""] + [n for n in score.get("notes") or [] if n]
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

    def bits(self, task: dict, texts: list[str], events: list[dict]) -> list[float | None]:
        """Per English text, the bits it saves the reader on the events (None when there are no events)."""
        if not events:
            return [None] * len(texts)
        prompt = task["prompts"][0]
        strings = self.tokens_of(prompt["token_ids"])
        top = repr(prompt["model_top"][0][0][0])

        def asks(eng: str) -> list[str]:
            head = f"An explanation of how the model computes this prediction:\n{eng}\n\n" if eng else ""
            return [ASK.format(top=top, tokens=repr(strings), english=head, position=e["position"], old=repr(strings[e["position"]]),
                               new=repr(self.tokens_of([e["new"]])[0])) for e in events]

        def loss(ps: list[float]) -> float:
            return sum(-math.log2(max(p if e["flipped"] else 1 - p, 1e-12)) for p, e in zip(ps, events)) / len(events)

        flat = asks("") + [q for t in texts for q in asks(t)]
        ps = self._p_yes(flat)
        n = len(events)
        base = loss(ps[:n])
        return [base - loss(ps[n * (1 + j): n * (2 + j)]) for j in range(len(texts))]
