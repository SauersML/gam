"""Induction: when a stretch of text repeats, the next token is the one that followed the current token in
its earlier copy ("A B ... A -> B"). A previous-token step writes, at each position, the token before it;
an induction step attends from the current token to the position whose previous token matches it and
copies the token there."""


def back(tokens):
    # each position attends to the one before it
    return [[t - 1] if t else [0] for t in range(len(tokens))]


def prev(tokens, back):
    # the token before each position
    return [tokens[js[0]] if t else None for t, js in enumerate(back)]


def agreement(tokens, j, t):
    # how many tokens before j agree with the tokens ending at t
    k = 0
    while k < j and tokens[j - 1 - k] == tokens[t - k]:
        k += 1
    return k


def match(tokens, prev):
    # each position attends to the earlier positions whose previous token is its own token, those whose
    # earlier context agrees longest with the current context
    out = []
    for t in range(len(tokens)):
        js = [j for j in range(t) if prev[j] == tokens[t]]
        best = max([agreement(tokens, j, t) for j in js] or [0])
        out.append([j for j in js if agreement(tokens, j, t) == best])
    return out


def answer(tokens, match):
    # the token at the latest such position: the one that followed the current token before
    return [tokens[js[-1]] if js else None for js in match]
