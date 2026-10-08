"""Number succession: a comma-separated run of numbers with a constant step continues with the last number
plus the step. The answer is what is left of " <next number>" after what is already written of it."""
from mech import align, claim


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


def step(numbers):
    # the difference between the last two complete numbers
    return [n[-1] - n[-2] if len(n) >= 2 else None for n in numbers]


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, numbers, step):
    # the number after the last complete one, or what is left of it once its first digits are written
    return [None if d is None else rest(" " + str(n[-1] + d), "".join(tokens[: t + 1]))
            for t, (n, d) in enumerate(zip(numbers, step))]
