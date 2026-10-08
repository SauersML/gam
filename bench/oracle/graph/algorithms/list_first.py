"""List indexing: after "List: a, b, c, d." the question asks for the first word, which is the word right
after the list's colon."""
from mech import bind, claim


def first(tokens):
    # the first word of the list: the token after the first colon
    out, seen = [], None
    for t, tok in enumerate(tokens):
        if seen is None and t > 0 and tokens[t - 1] == ":":
            seen = tok
        out.append(seen)
    return out


def answer(tokens, first):
    # the first word, asked for at the end
    return first
