"""Gerund: the -ing form of a verb: a final e drops (not ee), ie becomes y, and a final consonant after a
single vowel doubles in a one-syllable verb or a verb stressed on its last syllable."""

STRESSED_LAST = {"begin", "forget", "admit", "prefer", "occur", "refer", "regret", "commit", "permit", "control",
                 "upset", "submit", "omit", "transfer"}


def verb(tokens):
    # the verb shown last: the word before ":" at the end of a "verb: verbing" line, or after "to"
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        line = text.split("\n")[-1]
        if line.endswith(":") and line[:-1].isalpha():
            out.append(line[:-1])
        elif " to " in text:
            word = text.rsplit(" to ", 1)[1].split(".")[0].split()
            out.append(word[0] if word and word[0].isalpha() else None)
        else:
            out.append(None)
    return out


def syllables(word):
    return sum(1 for k, c in enumerate(word) if c in "aeiou" and (k == 0 or word[k - 1] not in "aeiou"))


def ing(word):
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee") and len(word) > 2:
        return word[:-1] + "ing"
    short = word[-1] not in "aeiouwxy" and word[-2:-1] in "aeiou" and word[-3:-2] not in "aeiou"
    if short and (syllables(word) == 1 or word in STRESSED_LAST):
        return word + word[-1] + "ing"
    return word + "ing"


def answer(tokens, verb):
    # its -ing form, after a space
    return [None if v is None else " " + ing(v) for v in verb]
