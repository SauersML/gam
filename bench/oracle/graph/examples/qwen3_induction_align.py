"""Induction on Qwen3-0.6B (behavior induction_random.words8): a list of 8 random words is repeated, and
at a point in the repeat the next token is the one that followed the same token in the first copy.

Notes from the behavior's 96 prompts (attention of each head at the target; removal = the head's output
columns zeroed, KL(M || M_e) in bits at the target): L1.H3, L2.H12, L15.H3 and L20.H0 put 61-86% of
their attention on the previous token (removing L20.H0 costs 0.26 bits); L3.H10, L16.H14 and L21.H8 put
71-85% on the token after the earlier copy of the current token, and L21.H8 matters most (0.61 bits).
Transcoders replace the MLPs; no MLP feature is named here.
"""
from mech import align, claim


def back(tokens):
    # each position attends to the one before it
    return [[t - 1] if t else [0] for t in range(len(tokens))]


def prev(tokens, back):
    # the token before each position
    return [tokens[js[0]] if t else None for t, js in enumerate(back)]


def match(tokens, prev):
    # each position attends to the earlier positions whose previous token is its own token
    return [[j for j in range(t) if prev[j] == tokens[t]] for t in range(len(tokens))]


def answer(tokens, match):
    # the token at the latest such position: the one that followed the current token before
    return [tokens[js[-1]] if js else None for js in match]


align(prev, <p:1.h.3>, <p:2.h.12>, <p:15.h.3>, <p:20.h.0>)
claim(back, <p:1.h.3>, <p:2.h.12>, <p:15.h.3>, <p:20.h.0>)
align(answer, <p:3.h.10>, <p:16.h.14>, <p:21.h.8>)
claim(match, <p:3.h.10>, <p:16.h.14>, <p:21.h.8>)
