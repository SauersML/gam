"""Comparative and superlative: an adjective takes -er or -est: a final e takes -r/-st, a consonant and y
become -ier/-iest, a short final consonant after one vowel doubles; good, bad and far are irregular."""

IRREGULAR = {"good": ("better", "best"), "bad": ("worse", "worst"), "far": ("farther", "farthest"),
             "many": ("more", "most"), "little": ("less", "least")}


def adjective(tokens):
    # the adjective asked about: before ":" on the last line, or in "This box is X, but ..."
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        line = text.split("\n")[-1]
        if " is " in line and "," in line:
            out.append(line.split(" is ", 1)[1].split(",")[0].strip())
        elif ":" in line and line.split(":")[0].isalpha():
            out.append(line.split(":")[0])
        else:
            out.append(None)
    return out


def superlative(tokens):
    # whether the superlative is asked for: -est examples, or "that box is the"
    return ["est\n" in "".join(tokens[: t + 1]) or " is the" in "".join(tokens[: t + 1]) for t in range(len(tokens))]


def degree(word, most):
    if word in IRREGULAR:
        return IRREGULAR[word][1 if most else 0]
    end = "est" if most else "er"
    if word.endswith("e"):
        return word + end[1:]
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "i" + end
    if word[-1] not in "aeiouwxy" and word[-2:-1] in "aeiou" and word[-3:-2] not in "aeiou":
        return word + word[-1] + end
    return word + end


def rest(full, text):
    # what is left of `full` after the longest beginning of it that the text ends with
    return full[max(n for n in range(len(full)) if text.endswith(full[:n])):]


def answer(tokens, adjective, superlative):
    # the adjective's -er or -est form, or what is left of it once its first tokens are written
    return [None if a is None else rest(" " + degree(a, s), "".join(tokens[: t + 1]))
            for t, (a, s) in enumerate(zip(adjective, superlative))]
