"""Case conversion: after examples, the new word in lower case (UPPER -> upper) or upper case."""
from mech import bind, claim


def word(tokens):
    # the word of the last line, before "->"
    out = []
    for t in range(len(tokens)):
        line = "".join(tokens[: t + 1]).split("\n")[-1]
        out.append(line.split("->")[0].strip() if "->" in line else None)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, word):
    # the word converted the other way, or what is left of it
    out = []
    for t, w in enumerate(word):
        text = "".join(tokens[: t + 1])
        out.append(None if not w else rest(" " + (w.lower() if w.isupper() else w.upper()), text.rsplit("->", 1)[1]))
    return out
