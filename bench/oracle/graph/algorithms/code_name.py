"""Python variable reuse: the code names again the variable it bound last: a function's parameter, a loop
variable, or a local just assigned. After a call's name it is the call's argument, after return the value."""
from mech import align, claim


def bound(tokens):
    # the name bound most recently: def f(NAME), for NAME in, NAME = ...
    out = []
    for t in range(len(tokens)):
        words = "".join(tokens[: t + 1]).replace("(", " ( ").replace(")", " ) ").replace(":", " : ").split()
        last = None
        for k, w in enumerate(words):
            if k >= 3 and words[k - 3] == "def" and words[k - 1] == "(" and w.isidentifier():
                last = w
            elif k >= 1 and words[k - 1] == "for" and w.isidentifier():
                last = w
            elif k + 1 < len(words) and words[k + 1] == "=" and w.isidentifier():
                last = w
        out.append(last)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, bound):
    # the name, as the argument of the call just named or the value just returned
    out = []
    for t, name in enumerate(bound):
        text = "".join(tokens[: t + 1])
        cut = max(text.rfind(w) for w in ("len", "print", "return"))
        if name is None or cut < 0:
            out.append(None)
        else:
            full = ("(" if text[cut:].startswith(("len", "print")) else " ") + name
            out.append(rest(full, text))
    return out
