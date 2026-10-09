"""Number patterns: consecutive squares (n*n) or triangular numbers (n(n+1)/2) continue with the next
one: a square's root grows by one; a triangular number's step grows by one."""


def numbers(tokens):
    # the complete numbers so far: runs of digits in the text that something other than a digit follows
    out = []
    for t in range(len(tokens)):
        text, nums, cur = "".join(tokens[: t + 1]), [], ""
        for c in text:
            if c.isdigit():
                cur += c
            elif cur:
                nums.append(int(cur))
                cur = ""
        out.append(nums)
    return out


def root(n):
    r = int(n ** 0.5)
    while r * r < n:
        r += 1
    return r if r * r == n else None


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, numbers):
    # the next square or triangular number, or what is left of it once its first digits are written
    out = []
    for t, n in enumerate(numbers):
        if len(n) < 2:
            out.append(None)
            continue
        square = root(n[-1]) is not None and root(n[-2]) == root(n[-1]) - 1
        nxt = (root(n[-1]) + 1) ** 2 if square else n[-1] + (n[-1] - n[-2]) + 1
        out.append(rest(" " + str(nxt), "".join(tokens[: t + 1])))
    return out
