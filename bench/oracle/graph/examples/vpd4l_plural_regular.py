"""Behavior plural.regular (vpd4l; M's top token is right on 47% of the targets): Plural of a regular
noun after a count above one. Phrasings: 'I have one {X} and you have two' | 'one {X}, two' | 'There
is one {X} here and three'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 3.10 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 3.09, layer 3 MLP 1.51, L3.H4 1.47, L2.H3 1.28, L2.H4 1.09, L3.H0 0.92,
L2.H2 0.65, layer 1 MLP 0.63, L3.H1 0.39, L3.H5 0.38, L2.H1 0.34 bits. The program lets every write
among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 3.09 bits
mlp1 = node(L[1].mlp)  # recovers 0.63 bits
h2_1 = node(L[2].head[1])  # recovers 0.34 bits
h2_2 = node(L[2].head[2])  # recovers 0.65 bits
h2_3 = node(L[2].head[3])  # recovers 1.28 bits
h2_4 = node(L[2].head[4])  # recovers 1.09 bits
h3_0 = node(L[3].head[0])  # recovers 0.92 bits
h3_1 = node(L[3].head[1])  # recovers 0.39 bits
h3_4 = node(L[3].head[4])  # recovers 1.47 bits
h3_5 = node(L[3].head[5])  # recovers 0.38 bits
mlp3 = node(L[3].mlp)  # recovers 1.51 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_4,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_1,
    h2_2 >> h3_4,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_1,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_1,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

