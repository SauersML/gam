"""Commonsense recall: the everyday tool used for a named task. The table is what the model recalls."""
from mech import bind, claim

TOOL = {
    'boil water': ' kettle', 'brush your teeth': ' toothbrush', 'call a friend': ' phone',
    'climb onto the roof': ' ladder', 'comb your hair': ' comb', 'cut paper': ' scissors',
    'cut wood': ' saw', 'dig a hole': ' shovel', 'drive a nail': ' hammer',
    'dry your hair': ' towel', 'eat soup': ' spoon', 'light a candle': ' match',
    'measure length': ' ruler', 'open a can': ' opener', 'paint a wall': ' brush',
    'paper': ' scissors', 'see in the dark': ' flashlight', 'sweep the floor': ' broom',
    'take a photo': ' camera', 'tell the time': ' watch', 'tighten a bolt': ' wrench',
    'unlock a door': ' key', 'water the plants': ' hose', 'wood': ' saw', 'write a letter': ' pen',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in TOOL if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the task's tool, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), TOOL[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
