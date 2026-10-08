"""Hypernym: the category a named thing belongs to. The table is what the model recalls."""
from mech import bind, claim

KIND = {
    'English': ' language', 'French': ' language', 'January': ' month', 'Jupiter': ' planet',
    'London': ' city', 'March': ' month', 'Mars': ' planet', 'Monday': ' day', 'Paris': ' city',
    'Spanish': ' language', 'Tokyo': ' city', 'Tuesday': ' day', 'Venus': ' planet',
    'ant': ' insect', 'apple': ' fruit', 'banana': ' fruit', 'basketball': ' sport',
    'bee': ' insect', 'beetle': ' insect', 'blue': ' color', 'bus': ' vehicle', 'car': ' vehicle',
    'carrot': ' vegetable', 'chess': ' game', 'cobra': ' snake', 'copper': ' metal',
    'daisy': ' flower', 'dog': ' animal', 'eagle': ' bird', 'elephant': ' animal',
    'gold': ' metal', 'green': ' color', 'guitar': ' instrument', 'hammer': ' tool',
    'iron': ' metal', 'jacket': ' clothing', 'lion': ' animal', 'mango': ' fruit',
    'maple': ' tree', 'oak': ' tree', 'onion': ' vegetable', 'parrot': ' bird',
    'piano': ' instrument', 'pine': ' tree', 'poker': ' game', 'potato': ' vegetable',
    'python': ' snake', 'red': ' color', 'rose': ' flower', 'salmon': ' fish', 'saw': ' tool',
    'shark': ' fish', 'shirt': ' clothing', 'soccer': ' sport', 'sock': ' clothing',
    'sparrow': ' bird', 'tennis': ' sport', 'trout': ' fish', 'truck': ' vehicle',
    'tulip': ' flower', 'violin': ' instrument', 'wrench': ' tool',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in KIND if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the thing's category, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), KIND[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
