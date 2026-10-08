"""Associative recall: after "word=number" pairs, "word=" is followed by that word's number."""
from mech import bind, claim


def pairs(tokens):
    # the word=number pairs written so far
    out = []
    for t in range(len(tokens)):
        found = {}
        for chunk in "".join(tokens[: t + 1]).replace(".", ",").split(","):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                if v.strip().isdigit():
                    found.setdefault(k.strip(), v.strip())
        out.append(found)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, pairs):
    # the number of the word just written before "=", or what is left of it
    out = []
    for t, found in enumerate(pairs):
        text = "".join(tokens[: t + 1])
        word = text.rsplit("=", 1)[0].replace(",", " ").replace(".", " ").split()[-1:] if "=" in text else []
        value = found.get(word[0]) if word else None
        out.append(None if value is None else rest(value, text.rsplit("=", 1)[1]))
    return out
