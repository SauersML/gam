"""Factual recall: the postal abbreviation of a named US state. The table is what the model recalls."""
from mech import align, claim

CODE = {
    'Alabama': ' AL', 'Alaska': ' AK', 'Arizona': ' AZ', 'Arkansas': ' AR', 'California': ' CA',
    'Colorado': ' CO', 'Florida': ' FL', 'Georgia': ' GA', 'Hawaii': ' HI', 'Idaho': ' ID',
    'Illinois': ' IL', 'Indiana': ' IN', 'Iowa': ' IA', 'Kansas': ' KS', 'Kentucky': ' KY',
    'Louisiana': ' LA', 'Maine': ' ME', 'Maryland': ' MD', 'Michigan': ' MI', 'Minnesota': ' MN',
    'Mississippi': ' MS', 'Missouri': ' MO', 'Montana': ' MT', 'Nebraska': ' NE', 'Nevada': ' NV',
    'Ohio': ' OH', 'Oklahoma': ' OK', 'Oregon': ' OR', 'Tennessee': ' TN', 'Texas': ' TX',
    'Utah': ' UT', 'Vermont': ' VT', 'Virginia': ' VA', 'Washington': ' WA', 'Wisconsin': ' WI',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in CODE if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the state's code, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), CODE[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
