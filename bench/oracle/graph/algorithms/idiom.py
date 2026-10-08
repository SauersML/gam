"""Fixed expression: the last word of a frequent multiword expression follows its first words. The
table is what the model recalls."""
from mech import align, claim

NEXT = {
    'a lot': ' of', 'accordance': ' with', 'according': ' to', 'again': ' again',
    'again and': ' again', 'all of': ' sudden', 'all of a': ' sudden', 'as a matter of': ' fact',
    'as far': ' as', 'as long': ' as', 'as soon': ' as', 'as well': ' as', 'at least': ' one',
    'at the same': ' time', 'back': ' forth', 'back and': ' forth', 'better late than': ' never',
    'black': ' white', 'black and': ' white', 'bread': ' butter', 'bread and': ' butter',
    'by': ' large', 'by and': ' large', 'by means': ' of', 'by the': ' way', 'due': ' to',
    'each': ' other', 'first come, first': ' served', 'for the first': ' time',
    'for the time': ' being', 'give or': ' take', 'here': ' there', 'here and': ' there',
    'in accordance': ' with', 'in addition': ' to', 'in case': ' of', 'in charge': ' of',
    'in favor': ' of', 'in front': ' of', 'in order': ' to', 'in spite': ' of', 'in terms': ' of',
    'in the long': ' run', 'in the middle': ' of', 'instead': ' of', 'ladies': ' gentlemen',
    'ladies and': ' gentlemen', 'last but not': ' least', 'law': ' order', 'long': ' run',
    'middle': ' of', 'more or': ' less', 'now': ' then', 'now and': ' then', 'odds': ' ends',
    'odds and': ' ends', 'on behalf': ' of', 'on the': ' contrary', 'on the other': ' hand',
    'on top': ' of', 'once upon': ' time', 'once upon a': ' time', 'over': ' over',
    'over and': ' over', 'peace': ' quiet', 'peace and': ' quiet', 'pros': ' cons',
    'pros and': ' cons', 'rather': ' than', 'rock': ' roll', 'rock and': ' roll', 'safe': ' sound',
    'safe and': ' sound', 'salt': ' pepper', 'salt and': ' pepper', 'sick': ' tired',
    'sick and': ' tired', 'so far': ' as', 'sooner or': ' later', 'spite': ' of', 'terms': ' of',
    'the': ' contrary', 'the United': ' States', 'time': ' again', 'time and': ' again',
    'top': ' of', 'trial': ' error', 'trial and': ' error', 'up': ' down', 'up and': ' down',
    'with respect': ' to',
}


def opening(tokens):
    # the longest table entry the text ends with
    out = []
    for t in range(len(tokens)):
        text = " " + "".join(tokens[: t + 1])
        found = [k for k in NEXT if text.endswith(" " + k)]
        out.append(max(found, key=len) if found else None)
    return out


def answer(tokens, opening):
    # the word that completes it
    return [None if k is None else NEXT[k] for k in opening]
