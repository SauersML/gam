"""Python variable reuse: the code names again the variable it bound last: a function's parameter, a loop
variable, or a local just assigned. After "(" the name follows with no space, elsewhere after a space."""
from mech import bind, claim


def bound(tokens):
    # the name bound most recently: def f(NAME, ...), for NAME in, NAME =
    out, last = [], None
    for t in range(len(tokens)):
        tok = tokens[t].strip()
        before = tokens[t - 1] if t else ""
        after = tokens[t + 1] if t + 1 < len(tokens) else None
        if before.strip() in ("(", ",") and any(tokens[k] == " def" or tokens[k] == "def" for k in range(t)) and \
                not any(x.strip() in (")", "):") for x in tokens[:t]):
            last = tok
        if before.strip() == "for":
            last = tok
        if after == " =" and tok.isidentifier():
            last = tok
        out.append(last)
    return out


def answer(tokens, bound):
    # that name, written straight after "(" or after a space
    return [None if name is None else name if tokens[t].endswith("(") else " " + name for t, name in enumerate(bound)]
