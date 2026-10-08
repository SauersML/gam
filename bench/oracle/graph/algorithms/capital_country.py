"""Factual recall in reverse: the country whose capital is the named city. The table is what the model recalls."""
from mech import bind, claim

COUNTRY = {
    'Abuja': ' Nigeria', 'Accra': ' Ghana', 'Amsterdam': ' Netherlands', 'Ankara': ' Turkey',
    'Athens': ' Greece', 'Baghdad': ' Iraq', 'Bangkok': ' Thailand', 'Beijing': ' China',
    'Beirut': ' Lebanon', 'Belgrade': ' Serbia', 'Berlin': ' Germany', 'Bern': ' Switzerland',
    'Bogota': ' Colombia', 'Brasilia': ' Brazil', 'Bratislava': ' Slovakia',
    'Brussels': ' Belgium', 'Bucharest': ' Romania', 'Budapest': ' Hungary', 'Cairo': ' Egypt',
    'Canberra': ' Australia', 'Caracas': ' Venezuela', 'Cardiff': ' Wales',
    'Copenhagen': ' Denmark', 'Dakar': ' Senegal', 'Damascus': ' Syria', 'Delhi': ' India',
    'Dodoma': ' Tanzania', 'Dublin': ' Ireland', 'Edinburgh': ' Scotland', 'Hanoi': ' Vietnam',
    'Havana': ' Cuba', 'Helsinki': ' Finland', 'Islamabad': ' Pakistan', 'Jakarta': ' Indonesia',
    'Jerusalem': ' Israel', 'Kabul': ' Afghanistan', 'Kampala': ' Uganda', 'Kathmandu': ' Nepal',
    'Khartoum': ' Sudan', 'Kingston': ' Jamaica', 'Kyiv': ' Ukraine', 'Lima': ' Peru',
    'Lisbon': ' Portugal', 'Madrid': ' Spain', 'Manila': ' Philippines', 'Montevideo': ' Uruguay',
    'Moscow': ' Russia', 'Nairobi': ' Kenya', 'Oslo': ' Norway', 'Ottawa': ' Canada',
    'Paris': ' France', 'Prague': ' Czechia', 'Quito': ' Ecuador', 'Rabat': ' Morocco',
    'Reykjavik': ' Iceland', 'Rome': ' Italy', 'Santiago': ' Chile', 'Seoul': ' Korea',
    'Sofia': ' Bulgaria', 'Stockholm': ' Sweden', 'Sucre': ' Bolivia', 'Tehran': ' Iran',
    'Tokyo': ' Japan', 'Tripoli': ' Libya', 'Vienna': ' Austria', 'Warsaw': ' Poland',
    'Zagreb': ' Croatia',
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
        found = [(whole(text, k) + len(k), len(k), k) for k in COUNTRY if whole(text, k) >= 0]
        out.append(max(found)[2] if found else None)
    return out


def answer(tokens, key):
    # the country of that capital, or the rest of it once its first tokens are written
    out = []
    for t, k in enumerate(key):
        if k is None:
            out.append(None)
            continue
        text, full = "".join(tokens[: t + 1]), COUNTRY[k]
        cut = max([n for n in range(1, len(full)) if text.endswith(full[:n])] or [0])
        out.append(full[cut:])
    return out
