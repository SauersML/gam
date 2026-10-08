"""Factual recall: the nationality adjective of the named country. The table is what the model recalls."""
from mech import bind, claim

DEMONYM = {
    'Afghanistan': ' Afghan', 'Argentina': ' Argentine', 'Australia': ' Australian',
    'Austria': ' Austrian', 'Belgium': ' Belgian', 'Bolivia': ' Bolivian', 'Brazil': ' Brazilian',
    'Bulgaria': ' Bulgarian', 'Canada': ' Canadian', 'Chile': ' Chilean', 'China': ' Chinese',
    'Colombia': ' Colombian', 'Croatia': ' Croatian', 'Cuba': ' Cuban', 'Denmark': ' Danish',
    'Ecuador': ' Ecuadorian', 'Egypt': ' Egyptian', 'Ethiopia': ' Ethiopian',
    'Finland': ' Finnish', 'France': ' French', 'Germany': ' German', 'Ghana': ' Ghanaian',
    'Greece': ' Greek', 'Hungary': ' Hungarian', 'Iceland': ' Icelandic', 'India': ' Indian',
    'Indonesia': ' Indonesian', 'Iran': ' Iranian', 'Iraq': ' Iraqi', 'Ireland': ' Irish',
    'Israel': ' Israeli', 'Italy': ' Italian', 'Jamaica': ' Jamaican', 'Japan': ' Japanese',
    'Kenya': ' Kenyan', 'Korea': ' Korean', 'Lebanon': ' Lebanese', 'Libya': ' Libyan',
    'Malaysia': ' Malaysian', 'Mexico': ' Mexican', 'Mongolia': ' Mongolian',
    'Morocco': ' Moroccan', 'Nepal': ' Nepalese', 'Netherlands': ' Dutch', 'Nigeria': ' Nigerian',
    'Norway': ' Norwegian', 'Pakistan': ' Pakistani', 'Peru': ' Peruvian',
    'Philippines': ' Filipino', 'Poland': ' Polish', 'Portugal': ' Portuguese',
    'Romania': ' Romanian', 'Russia': ' Russian', 'Scotland': ' Scottish', 'Serbia': ' Serbian',
    'Spain': ' Spanish', 'Sudan': ' Sudanese', 'Sweden': ' Swedish', 'Switzerland': ' Swiss',
    'Syria': ' Syrian', 'Tanzania': ' Tanzanian', 'Thailand': ' Thai', 'Turkey': ' Turkish',
    'Uganda': ' Ugandan', 'Ukraine': ' Ukrainian', 'Venezuela': ' Venezuelan',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in DEMONYM if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the nationality adjective of that country, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), DEMONYM[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
