"""Object tracking: "X put the A in the P and the B in the Q. The A is in the" -> P: the place named three
tokens after the earlier mention of the object asked about."""
from mech import bind, claim


def asked(tokens):
    # the object the question names: the token two before "in the" at the end
    return [tokens[t - 3] if t >= 3 and tokens[t - 1] == " in" else None for t in range(len(tokens))]


def answer(tokens, asked):
    # the place after "<object> in the" at the object's first mention
    out = []
    for t, obj in enumerate(asked):
        js = [j for j in range(t - 3) if tokens[j] == obj and j + 3 < t]
        out.append(tokens[js[0] + 3] if js else None)
    return out
