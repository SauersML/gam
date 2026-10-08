"""Word succession: in a run of ordinal ("fourth, fifth") or cardinal ("four, five") words the next word is
the next one of the same list."""
from mech import bind, claim

LISTS = [["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
          "eleventh", "twelfth"],
         ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]]


def word(tokens):
    # the latest number word: (its list, its rank)
    out, last = [], None
    for t, tok in enumerate(tokens):
        w = tok.strip().lower()
        pronoun = w == "one" and t and tokens[t - 1].strip() in ("next", "the", "this")  # "the next one"
        for k, words in enumerate(LISTS):
            if w in words and not pronoun:
                last = (k, words.index(w))
        out.append(last)
    return out


def answer(tokens, word):
    # the next word of that list, after a space
    return [None if w is None or w[1] + 1 >= len(LISTS[w[0]]) else " " + LISTS[w[0]][w[1] + 1] for w in word]
