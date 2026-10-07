"""Behavior sva.simple (qwen3-0.6b; M's top token is right on 6% of the targets): Subject-verb
agreement after a determiner and noun; the counterfactual flips the subject's number.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 4.22 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 6.75 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: L0.H3 3.76, L24.H1 0.94 bits. The program lets every write among them reach every later
read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 3.76 bits
h24_1 = node(L[24].head[1])  # recovers 0.94 bits

edges(
    embed >> h0_3,
    embed >> h24_1,
    h0_3 >> h24_1,
    h0_3 >> logits,
    h24_1 >> logits,
)

