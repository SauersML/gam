"""Quote closing: inside an open quotation that ends a sentence, the next token closes it with '."'."""
from mech import align, claim


def inside(tokens):
    # whether a quotation is open: an odd number of quote marks so far
    out, n = [], 0
    for tok in tokens:
        n += tok.count('"')
        out.append(n % 2 == 1)
    return out


def answer(tokens, inside):
    # the period and closing quote
    return ['."' if q else None for q in inside]
