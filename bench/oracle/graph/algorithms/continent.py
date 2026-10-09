"""Factual recall: the continent of a named country. The table is what the model recalls."""

CONTINENT = {
    'Afghanistan': ' Asia', 'Argentina': ' South America', 'Australia': ' Australia',
    'Austria': ' Europe', 'Belgium': ' Europe', 'Bolivia': ' South America',
    'Brazil': ' South America', 'Bulgaria': ' Europe', 'Canada': ' North America',
    'Chile': ' South America', 'China': ' Asia', 'Colombia': ' South America',
    'Croatia': ' Europe', 'Cuba': ' North America', 'Czechia': ' Europe', 'Denmark': ' Europe',
    'Ecuador': ' South America', 'Egypt': ' Africa', 'Ethiopia': ' Africa', 'Finland': ' Europe',
    'France': ' Europe', 'Germany': ' Europe', 'Ghana': ' Africa', 'Greece': ' Europe',
    'Hungary': ' Europe', 'Iceland': ' Europe', 'India': ' Asia', 'Indonesia': ' Asia',
    'Iran': ' Asia', 'Iraq': ' Asia', 'Ireland': ' Europe', 'Israel': ' Asia', 'Italy': ' Europe',
    'Jamaica': ' North America', 'Japan': ' Asia', 'Kenya': ' Africa', 'Korea': ' Asia',
    'Lebanon': ' Asia', 'Libya': ' Africa', 'Malaysia': ' Asia', 'Mexico': ' North America',
    'Mongolia': ' Asia', 'Morocco': ' Africa', 'Nepal': ' Asia', 'Netherlands': ' Europe',
    'Nigeria': ' Africa', 'Norway': ' Europe', 'Pakistan': ' Asia', 'Peru': ' South America',
    'Philippines': ' Asia', 'Poland': ' Europe', 'Portugal': ' Europe', 'Romania': ' Europe',
    'Russia': ' Europe', 'Scotland': ' Europe', 'Senegal': ' Africa', 'Serbia': ' Europe',
    'Slovakia': ' Europe', 'Spain': ' Europe', 'Sudan': ' Africa', 'Sweden': ' Europe',
    'Switzerland': ' Europe', 'Syria': ' Asia', 'Tanzania': ' Africa', 'Thailand': ' Asia',
    'Turkey': ' Asia', 'Uganda': ' Africa', 'Ukraine': ' Europe', 'Uruguay': ' South America',
    'Venezuela': ' South America', 'Vietnam': ' Asia', 'Wales': ' Europe',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in CONTINENT if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the country's continent, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), CONTINENT[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
