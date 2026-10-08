"""Alphabet succession: in a run of letters the next letter follows the current one."""
from mech import bind, claim


def letter(tokens):
    # the latest single letter of the run
    out, last = [], None
    for tok in tokens:
        if len(tok.strip()) == 1 and tok.strip().isalpha():
            last = tok.strip()
        out.append(last)
    return out


def answer(tokens, letter):
    # the next letter, after a space
    return [None if c is None or c in "zZ" else " " + chr(ord(c) + 1) for c in letter]
