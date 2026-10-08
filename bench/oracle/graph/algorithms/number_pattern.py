"""Number patterns: consecutive squares (n*n) or triangular numbers (n(n+1)/2) continue with the next
one: a square's root grows by one; a triangular number's step grows by one."""
from mech import bind, claim


def numbers(tokens):
    # the numbers written so far (digit tokens joined)
    out, nums, cur = [], [], ""
    for tok in tokens:
        s = tok.strip()
        if s.isdigit():
            cur = cur + s if cur and not tok.startswith(" ") else s
        elif cur:
            nums.append(int(cur))
            cur = ""
        out.append(list(nums))
    return out


def root(n):
    r = int(n ** 0.5)
    while r * r < n:
        r += 1
    return r if r * r == n else None


def answer(tokens, numbers):
    # after a comma, the next square or triangular number
    out = []
    for t, n in enumerate(numbers):
        if len(n) < 2 or tokens[t] != ",":
            out.append(None)
        elif root(n[-1]) is not None and root(n[-2]) == root(n[-1]) - 1:
            out.append(" " + str((root(n[-1]) + 1) ** 2))
        else:
            out.append(" " + str(n[-1] + (n[-1] - n[-2]) + 1))
    return out
