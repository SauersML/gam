"""Spelling: 'The word "wallet" starts with the letter' -> " W": the quoted word's first letter, capital."""
from mech import align, claim


def word(tokens):
    # the quoted word: the text between the first two quote marks
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        parts = text.split('"')
        out.append(parts[1].strip() if len(parts) >= 3 else None)
    return out


def answer(tokens, word):
    # its first letter as a capital, after a space
    return [" " + w[0].upper() if w else None for w in word]
