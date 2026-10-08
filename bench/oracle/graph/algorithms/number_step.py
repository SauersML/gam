"""Number succession: a comma-separated run of numbers with a constant step continues with the last number
plus the step. A number may span several tokens: after its first tokens, the answer is the rest of it."""
from mech import bind, claim


def partial(tokens):
    # the digits of the number being written at each position ("" when none is)
    out, cur = [], ""
    for t, tok in enumerate(tokens):
        s = tok.strip()
        cur = (cur + s if cur and not tok.startswith(" ") else s) if s.isdigit() else ""
        out.append(cur)
    return out


def numbers(tokens, partial):
    # the complete numbers before each position's partial one
    out, nums = [], []
    for t in range(len(tokens)):
        if t and partial[t - 1] and partial[t] != partial[t - 1] + tokens[t].strip():
            nums.append(int(partial[t - 1]))
        out.append(list(nums))
    return out


def step(numbers):
    # the difference between the last two complete numbers
    return [n[-1] - n[-2] if len(n) >= 2 else None for n in numbers]


def answer(tokens, partial, numbers, step):
    # the next number, or the rest of it once its first tokens are written
    out = []
    for t in range(len(tokens)):
        n, d, p = numbers[t], step[t], partial[t]
        if d is None:
            out.append(None)
        elif not p:
            out.append(" " + str(n[-1] + d))
        else:
            nxt = str(n[-1] + d)
            out.append(nxt[len(p):] if nxt.startswith(p) and len(nxt) > len(p) else None)
    return out
