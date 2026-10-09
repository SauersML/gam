"""Number comparison: the larger of two numbers ("Which number is larger, 67 or 52?") or the smallest of
several ("The smallest of 71, 49 and 65 is")."""


def numbers(tokens):
    # the numbers of the question: the complete numbers written before its last word
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


def smallest(tokens):
    # whether the question asks for the smallest
    return ["smallest" in "".join(tokens[: t + 1]) for t in range(len(tokens))]


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, numbers, smallest):
    # the smallest or the largest, or what is left of it once its first digits are written
    out = []
    for t, (n, low) in enumerate(zip(numbers, smallest)):
        if len(n) < 2:
            out.append(None)
            continue
        pick = min(n) if low else max(n[:2])
        out.append(rest(" " + str(pick), "".join(tokens[: t + 1])))
    return out
