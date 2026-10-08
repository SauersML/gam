"""Ordinal succession: in a run of ordinal words the next word is the next ordinal."""
from mech import bind, claim

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "eleventh", "twelfth"]


def ordinal(tokens):
    # the rank of the latest ordinal word
    out, last = [], None
    for tok in tokens:
        if tok.strip() in ORDINALS:
            last = ORDINALS.index(tok.strip())
        out.append(last)
    return out


def answer(tokens, ordinal):
    # the next ordinal, after a space
    return [None if k is None or k + 1 >= len(ORDINALS) else " " + ORDINALS[k + 1] for k in ordinal]
