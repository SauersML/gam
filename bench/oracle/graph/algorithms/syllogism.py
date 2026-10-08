"""Syllogism: "All X are Ys. N is one of the X. Therefore N is a" -> " Y": the category, singular."""
from mech import align, claim


def category(tokens):
    # the word after the first " are": the category, plural
    out, cat = [], None
    for t, tok in enumerate(tokens):
        if cat is None and t and tokens[t - 1] == " are":
            cat = tok
        out.append(cat)
    return out


def answer(tokens, category):
    # the category without its plural "s"
    return [c[:-1] if c and c.endswith("s") else c for c in category]
