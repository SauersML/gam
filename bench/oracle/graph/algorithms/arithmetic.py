"""Arithmetic: after worked examples, the result of the last "a op b =" problem."""
from mech import bind, claim


def problem(tokens):
    # the last problem in the text: (a, op, b), once its "=" is written
    out = []
    for t in range(len(tokens)):
        line = "".join(tokens[: t + 1]).split("\n")[-1]
        parts = line.split("=")[0].split() if "=" in line else []
        ok = len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit() and parts[1] in "+-*"
        out.append((int(parts[0]), parts[1], int(parts[2])) if ok else None)
    return out


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, problem):
    # the result, or what is left of it once its first digits are written
    out = []
    for t, p in enumerate(problem):
        if p is None:
            out.append(None)
            continue
        a, op, b = p
        value = a + b if op == "+" else a - b if op == "-" else a * b
        out.append(rest(" " + str(value), "".join(tokens[: t + 1])))
    return out
