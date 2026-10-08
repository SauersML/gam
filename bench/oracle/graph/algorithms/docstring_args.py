"""Docstring arguments: after documenting the first arguments of a function, the next documented name is
the next argument of its signature."""
from mech import bind, claim


def params(tokens):
    # the signature's arguments, in order: the names between "(" and ")"
    out, names, open_ = [], [], False
    for t, tok in enumerate(tokens):
        s = tok.strip()
        if s == "(" and t and any(x.strip() == "def" for x in tokens[:t]) and not names:
            open_ = True
        elif open_ and s.startswith(")"):
            open_ = False
        elif open_ and s not in (",", ""):
            names.append(s)
        out.append(list(names))
    return out


def documented(tokens):
    # the names documented so far: a name followed by ":" after "Args"
    out, names, args = [], [], False
    for t, tok in enumerate(tokens):
        if tok.strip() == "Args":
            args = True
        elif args and tok == ":" and t and tokens[t - 1].strip():
            names.append(tokens[t - 1].strip())
        out.append(list(names))
    return out


def answer(tokens, params, documented):
    # the first argument not documented yet, at the start of a new line
    out = []
    for t in range(len(tokens)):
        rest = [p for p in params[t] if p not in documented[t]]
        out.append(rest[0] if rest and tokens[t].strip() == "" and "Args" in "".join(tokens[: t + 1]) else None)
    return out
