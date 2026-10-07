"""Behavior name_repeat.story (vpd4l; M's top token is right on 83% of the targets): Name repetition in
prose: a person introduced by full name is later mentioned by first name and the surname follows.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 16.99 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 16.98, L2.H4 14.31, layer 3 MLP 11.38, L2.H3 7.13, L3.H5 6.56, layer 1
MLP 4.32, L1.H1 3.30, layer 2 MLP 3.06 bits. The program lets every write among them reach every
later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 16.98 bits
h1_1 = node(L[1].head[1])  # recovers 3.30 bits
mlp1 = node(L[1].mlp)  # recovers 4.32 bits
h2_3 = node(L[2].head[3])  # recovers 7.13 bits
h2_4 = node(L[2].head[4])  # recovers 14.31 bits
mlp2 = node(L[2].mlp)  # recovers 3.06 bits
h3_5 = node(L[3].head[5])  # recovers 6.56 bits
mlp3 = node(L[3].mlp)  # recovers 11.38 bits

edges(
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_3,
    h1_1 >> h2_4,
    h1_1 >> mlp2,
    h1_1 >> h3_5,
    h1_1 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

