"""Factual recall: the atomic number of a named element. The table is what the model recalls."""
from mech import bind, claim

NUMBER = {
    'aluminum': ' 13', 'argon': ' 18', 'beryllium': ' 4', 'boron': ' 5', 'calcium': ' 20',
    'carbon': ' 6', 'chlorine': ' 17', 'copper': ' 29', 'fluorine': ' 9', 'gold': ' 79',
    'helium': ' 2', 'hydrogen': ' 1', 'iodine': ' 53', 'iron': ' 26', 'lead': ' 82',
    'lithium': ' 3', 'magnesium': ' 12', 'mercury': ' 80', 'neon': ' 10', 'nickel': ' 28',
    'nitrogen': ' 7', 'oxygen': ' 8', 'phosphorus': ' 15', 'platinum': ' 78', 'potassium': ' 19',
    'silicon': ' 14', 'silver': ' 47', 'sodium': ' 11', 'sulfur': ' 16', 'tin': ' 50',
    'uranium': ' 92', 'zinc': ' 30',
}


def whole(text, k):
    # where k last occurs in text as whole words (no letter just before or after it), else -1
    at = text.rfind(k)
    while at >= 0 and ((at and text[at - 1].isalpha()) or text[at + len(k): at + len(k) + 1].isalpha()):
        at = text.rfind(k, 0, at)
    return at


def key(tokens):
    # the table's entry the text names last (the longest, when entries overlap)
    out = []
    for t in range(len(tokens)):
        text = "".join(tokens[: t + 1])
        found = [(whole(text, k) + len(k), len(k), k) for k in NUMBER if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the element's atomic number, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), NUMBER[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
