"""Factual recall: the chemical symbol of a named element. The table is what the model recalls."""

SYMBOL = {
    'aluminum': ' Al', 'argon': ' Ar', 'boron': ' B', 'calcium': ' Ca', 'carbon': ' C',
    'chlorine': ' Cl', 'chromium': ' Cr', 'cobalt': ' Co', 'copper': ' Cu', 'fluorine': ' F',
    'gold': ' Au', 'helium': ' He', 'hydrogen': ' H', 'iodine': ' I', 'iron': ' Fe', 'lead': ' Pb',
    'lithium': ' Li', 'magnesium': ' Mg', 'manganese': ' Mn', 'mercury': ' Hg', 'neon': ' Ne',
    'nickel': ' Ni', 'nitrogen': ' N', 'oxygen': ' O', 'phosphorus': ' P', 'platinum': ' Pt',
    'potassium': ' K', 'silicon': ' Si', 'silver': ' Ag', 'sodium': ' Na', 'sulfur': ' S',
    'tin': ' Sn', 'titanium': ' Ti', 'tungsten': ' W', 'uranium': ' U', 'zinc': ' Zn',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in SYMBOL if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the element's symbol, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), SYMBOL[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
