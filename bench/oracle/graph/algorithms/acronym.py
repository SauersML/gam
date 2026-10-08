"""Acronym: after a capitalized three-word name and "(", the next tokens spell its initials."""
from mech import bind, claim


def initials(tokens):
    # the initials of the capitalized words before the latest "("
    out = []
    for t in range(len(tokens)):
        opened = [j for j in range(t + 1) if tokens[j].endswith("(")]
        words = [x.strip() for x in tokens[: opened[-1]] if x.startswith(" ") and x.strip()[:1].isupper()] if opened else []
        out.append("".join(w[0] for w in words[-3:]) if opened else None)
    return out


def answer(tokens, initials):
    # the initials not written yet since the "("
    out = []
    for t, letters in enumerate(initials):
        if letters is None:
            out.append(None)
            continue
        opened = max(j for j in range(t + 1) if tokens[j].endswith("("))
        written = "".join(tokens[opened + 1: t + 1])
        out.append(letters[len(written):] if letters.startswith(written) and len(letters) > len(written) else None)
    return out
