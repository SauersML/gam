"""Behavior acronym.three_word (qwen3-0.6b; M's top token is right on 90% of the targets): Acronym:
after a three-word capitalized name and an open parenthesis, the next tokens spell its initials.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 11.48 bits per target token between the clean and the counterfactual next-token
distributions, and more than their opaque price per token (1/2 log2 N bits per weight, N = 2^24):
0.28 for a head, 6.75 for a whole MLP; an MLP may instead enter as the neurons that pay for
themselves: L22.H13 4.29, L22.H12 3.12, L11.H9 2.98, L22.H9 2.15, L27.H15 1.95, L0.H12 1.42, L0.H3
1.32, L25.H9 1.19 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

h0_3 = node(L[0].head[3])  # recovers 1.32 bits
h0_12 = node(L[0].head[12])  # recovers 1.42 bits
h11_9 = node(L[11].head[9])  # recovers 2.98 bits
h22_9 = node(L[22].head[9])  # recovers 2.15 bits
h22_12 = node(L[22].head[12])  # recovers 3.12 bits
h22_13 = node(L[22].head[13])  # recovers 4.29 bits
h25_9 = node(L[25].head[9])  # recovers 1.19 bits
h27_15 = node(L[27].head[15])  # recovers 1.95 bits

edges(
    embed >> h0_3,
    embed >> h0_12,
    embed >> h11_9,
    embed >> h22_9,
    embed >> h22_12,
    embed >> h22_13,
    embed >> h25_9,
    embed >> h27_15,
    h0_3 >> h11_9,
    h0_3 >> h22_9,
    h0_3 >> h22_12,
    h0_3 >> h22_13,
    h0_3 >> h25_9,
    h0_3 >> h27_15,
    h0_12 >> h11_9,
    h0_12 >> h22_9,
    h0_12 >> h22_12,
    h0_12 >> h22_13,
    h0_12 >> h25_9,
    h0_12 >> h27_15,
    h11_9 >> h22_9,
    h11_9 >> h22_12,
    h11_9 >> h22_13,
    h11_9 >> h25_9,
    h11_9 >> h27_15,
    h22_9 >> h25_9,
    h22_9 >> h27_15,
    h22_12 >> h25_9,
    h22_12 >> h27_15,
    h22_13 >> h25_9,
    h22_13 >> h27_15,
    h25_9 >> h27_15,
    h0_3 >> logits,
    h0_12 >> logits,
    h11_9 >> logits,
    h22_9 >> logits,
    h22_12 >> logits,
    h22_13 >> logits,
    h25_9 >> logits,
    h27_15 >> logits,
)

