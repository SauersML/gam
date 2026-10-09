"""Antonym: the opposite of a common word. The table is what the model recalls."""

OPPOSITE = {
    'above': ' below', 'accept': ' reject', 'alive': ' dead', 'always': ' never',
    'ancient': ' modern', 'arrive': ' leave', 'asleep': ' awake', 'before': ' after',
    'begin': ' end', 'big': ' small', 'black': ' white', 'buy': ' sell', 'cheap': ' expensive',
    'clean': ' dirty', 'day': ' night', 'deep': ' shallow', 'early': ' late', 'east': ' west',
    'easy': ' difficult', 'fast': ' slow', 'first': ' last', 'friend': ' enemy', 'full': ' empty',
    'give': ' take', 'good': ' bad', 'happy': ' sad', 'hard': ' soft', 'heavy': ' light',
    'high': ' low', 'hot': ' cold', 'increase': ' decrease', 'inside': ' outside',
    'left': ' right', 'light': ' dark', 'long': ' short', 'loud': ' quiet', 'love': ' hate',
    'major': ' minor', 'male': ' female', 'maximum': ' minimum', 'near': ' far', 'noisy': ' quiet',
    'north': ' south', 'old': ' young', 'open': ' closed', 'polite': ' rude',
    'positive': ' negative', 'possible': ' impossible', 'push': ' pull', 'question': ' answer',
    'rich': ' poor', 'safe': ' dangerous', 'smooth': ' rough', 'strong': ' weak', 'sweet': ' sour',
    'tall': ' short', 'thick': ' thin', 'true': ' false', 'up': ' down', 'upper': ' lower',
    'visible': ' invisible', 'wet': ' dry', 'wide': ' narrow', 'win': ' lose',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in OPPOSITE if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the word's opposite, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), OPPOSITE[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
