"""Past tense: an irregular verb's past form is recalled; a regular verb adds -ed (with -d after e, -ied
after a consonant and y, and a doubled final consonant after a short vowel)."""

IRREGULAR = {
    'break': ' broke', 'bring': ' brought', 'buy': ' bought', 'catch': ' caught',
    'drink': ' drank', 'drive': ' drove', 'eat': ' ate', 'feel': ' felt', 'find': ' found',
    'fly': ' flew', 'give': ' gave', 'go': ' went', 'leave': ' left', 'lose': ' lost',
    'make': ' made', 'ride': ' rode', 'run': ' ran', 'see': ' saw', 'sell': ' sold',
    'sing': ' sang', 'sit': ' sat', 'sleep': ' slept', 'speak': ' spoke', 'stand': ' stood',
    'take': ' took', 'teach': ' taught', 'tell': ' told', 'think': ' thought', 'win': ' won',
    'write': ' wrote',
}


def verb(tokens):
    # the verb of the present-tense sentence: the word after its subject ("I", "they", "we", ...)
    out, last = [], None
    for t, tok in enumerate(tokens):
        subject = [x.strip() for x in tokens[:t] if x.strip() not in ("usually", "always", "often", "normally")]
        if subject and subject[-1] in ("I", "they", "we", "you") and tok.strip().isalpha() and \
                tok.strip() not in ("usually", "always", "often", "normally"):
            last = last if last is not None else tok.strip()
        out.append(last)
    return out


def regular(word):
    if word.endswith("e"):
        return word + "d"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ied"
    if len(word) <= 4 and word[-1] not in "aeiouwxy" and word[-2:-1] in "aeiou" and word[-3:-2] not in "aeiou":
        return word + word[-1] + "ed"
    return word + "ed"


def answer(tokens, verb):
    # its past form, after a space
    return [None if v is None else IRREGULAR[v] if v in IRREGULAR else " " + regular(v) for v in verb]
