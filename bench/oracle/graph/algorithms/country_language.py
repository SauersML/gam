"""Factual recall: the main language of the named country. The table is what the model recalls."""
from mech import bind, claim

LANGUAGE = {
    'Afghanistan': ' Pashto', 'Argentina': ' Spanish', 'Australia': ' English',
    'Austria': ' German', 'Belgium': ' Dutch', 'Bolivia': ' Spanish', 'Brazil': ' Portuguese',
    'Bulgaria': ' Bulgarian', 'Canada': ' English', 'Chile': ' Spanish', 'China': ' Chinese',
    'Colombia': ' Spanish', 'Croatia': ' Croatian', 'Cuba': ' Spanish', 'Czechia': ' Czech',
    'Denmark': ' Danish', 'Ecuador': ' Spanish', 'Egypt': ' Arabic', 'Ethiopia': ' Amharic',
    'Finland': ' Finnish', 'France': ' French', 'Germany': ' German', 'Ghana': ' English',
    'Greece': ' Greek', 'Hungary': ' Hungarian', 'Iceland': ' Icelandic', 'India': ' Hindi',
    'Indonesia': ' Indonesian', 'Iran': ' Persian', 'Iraq': ' Arabic', 'Ireland': ' English',
    'Israel': ' Hebrew', 'Italy': ' Italian', 'Jamaica': ' English', 'Japan': ' Japanese',
    'Kenya': ' Swahili', 'Korea': ' Korean', 'Lebanon': ' Arabic', 'Libya': ' Arabic',
    'Malaysia': ' Malay', 'Mexico': ' Spanish', 'Mongolia': ' Mongolian', 'Morocco': ' Arabic',
    'Nepal': ' Nepali', 'Netherlands': ' Dutch', 'Nigeria': ' English', 'Norway': ' Norwegian',
    'Pakistan': ' Urdu', 'Peru': ' Spanish', 'Philippines': ' Filipino', 'Poland': ' Polish',
    'Portugal': ' Portuguese', 'Romania': ' Romanian', 'Russia': ' Russian',
    'Scotland': ' English', 'Senegal': ' French', 'Serbia': ' Serbian', 'Slovakia': ' Slovak',
    'Spain': ' Spanish', 'Sudan': ' Arabic', 'Sweden': ' Swedish', 'Switzerland': ' German',
    'Syria': ' Arabic', 'Tanzania': ' Swahili', 'Thailand': ' Thai', 'Turkey': ' Turkish',
    'Uganda': ' English', 'Ukraine': ' Ukrainian', 'Uruguay': ' Spanish', 'Venezuela': ' Spanish',
    'Vietnam': ' Vietnamese', 'Wales': ' Welsh',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in LANGUAGE if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the language of that country, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), LANGUAGE[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
