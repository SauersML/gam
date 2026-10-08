"""Bracket closing: after a call's last argument, the next token closes every call still open."""
from mech import bind, claim


def depth(tokens):
    # how many "(" are still open
    out, d = [], 0
    for tok in tokens:
        d += tok.count("(") - tok.count(")")
        out.append(d)
    return out


def answer(tokens, depth):
    # one ")" per open call
    return [")" * d if d > 0 else None for d in depth]
