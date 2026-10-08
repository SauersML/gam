"""Induction on vpd4l: when a stretch of text repeats, predict the token that followed the current token
in its first copy ("A B ... A -> B").

Notes from measurements on repeated Pile text (removing one subcomponent, W - u_i v_i^T at every
token, and the KL(M || M_e) it costs per target token): layer 1's query 316 and key 329 cost the most
(3.8 bits each), so layer 1 holds the previous-token step; layer 2's key 206 (0.45) and output 735 (0.69)
the induction step; layer 3's output 806 (0.35) writes the copied token again. Value subcomponents cost
less each (0.07 to 0.25); the ones named are the largest of their layer.
"""
from mech import align, claim


def back(tokens):
    # each position attends to the one before it
    return [[t - 1] if t else [0] for t in range(len(tokens))]


def prev(tokens, back):
    # the token before each position
    return [tokens[js[0]] if t else None for t, js in enumerate(back)]


def match(tokens, prev):
    # each position attends to the earlier positions whose previous token is its own token:
    # the positions right after the earlier copies of the current token
    return [[j for j in range(t) if prev[j] == tokens[t]] for t in range(len(tokens))]


def answer(tokens, match):
    # the token at the latest such position: the one that followed the current token before
    return [tokens[js[-1]] if js else None for js in match]


claim(back, <p:1.q.316>, <p:1.k.329>)
align(prev, <p:1.v.228>, <p:1.v.346>, <p:1.o.311>, <p:1.o.340>)
claim(match, <p:2.q.335>, <p:2.k.206>)
align(answer, <p:2.v.559>, <p:2.o.735>, <p:3.v.677>, <p:3.o.806>)
