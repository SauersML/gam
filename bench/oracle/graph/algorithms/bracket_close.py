"""Bracket closing: after the last item, the next token closes every bracket still open, innermost first."""
from mech import align, claim

CLOSE = {"(": ")", "[": "]", "{": "}"}


def open_brackets(tokens):
    # the brackets still open in the text, outermost first
    out = []
    for t in range(len(tokens)):
        stack = []
        for c in "".join(tokens[: t + 1]):
            if c in CLOSE:
                stack.append(c)
            elif c in CLOSE.values() and stack and CLOSE[stack[-1]] == c:
                stack.pop()
        out.append(stack)
    return out


def answer(tokens, open_brackets):
    # their closing marks, innermost first
    return ["".join(CLOSE[b] for b in reversed(s)) if s else None for s in open_brackets]
