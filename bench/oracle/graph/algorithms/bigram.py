"""Frequent bigram: the second word of a common multiword name follows its first word. The table is what the model recalls."""

SECOND = {
    'Abraham': ' Lincoln', 'Abu': ' Dhabi', 'Air': ' Force', 'Albert': ' Einstein',
    'Angela': ' Merkel', 'Atlantic': ' Ocean', 'Bank': ' of', 'Barack': ' Obama', 'Bill': ' Gates',
    'Bob': ' Dylan', 'Buenos': ' Aires', 'Burger': ' King', 'Charles': ' Darwin', 'Coca': ' Cola',
    'Costa': ' Rica', 'Donald': ' Trump', 'Elon': ' Musk', 'Elvis': ' Presley',
    'George': ' Washington', 'Harry': ' Potter', 'Hillary': ' Clinton', 'Hong': ' Kong',
    'Isaac': ' Newton', 'Kuala': ' Lumpur', 'Lady': ' Gaga', 'Las': ' Vegas', 'Los': ' Angeles',
    'Manchester': ' United', 'Martin': ' Luther', 'Michael': ' Jackson', 'Middle': ' East',
    'Mother': ' Teresa', 'Mount': ' Everest', 'Nelson': ' Mandela', 'New': ' York',
    'Notre': ' Dame', 'Pacific': ' Ocean', 'Pearl': ' Harbor', 'Pink': ' Floyd',
    'Prime': ' Minister', 'Puerto': ' Rico', 'Real': ' Madrid', 'Red': ' Cross', 'Rio': ' de',
    'Rolling': ' Stones', 'San': ' Francisco', 'Saudi': ' Arabia', 'Silicon': ' Valley',
    'Sri': ' Lanka', 'Star': ' Wars', 'Steve': ' Jobs', 'Supreme': ' Court', 'Taylor': ' Swift',
    'Tel': ' Aviv', 'United': ' States', 'Vice': ' President', 'Vladimir': ' Putin',
    'Wall': ' Street', 'White': ' House', 'Wikipedia': ' article', 'William': ' Shakespeare',
    'Winston': ' Churchill', 'World': ' War',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in SECOND if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the name's second word, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), SECOND[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
