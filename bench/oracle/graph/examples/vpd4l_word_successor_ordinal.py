"""Behavior word_successor.ordinal (vpd4l; M's top token is right on 54% of the targets): Ordinal
succession: a run of ordinal words; the next word continues the order.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.95 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.94, L2.H3 1.21, layer 3 MLP 0.88, L2.H4 0.69, L3.H0 0.57, layer 1 MLP
0.46, L3.H5 0.41, L2.H1 0.33, L2.H2 0.29 bits. The program lets every write among them reach every
later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 1.94 bits
mlp1 = node(L[1].mlp)  # recovers 0.46 bits
h2_1 = node(L[2].head[1])  # recovers 0.33 bits
h2_2 = node(L[2].head[2])  # recovers 0.29 bits
h2_3 = node(L[2].head[3])  # recovers 1.21 bits
h2_4 = node(L[2].head[4])  # recovers 0.69 bits
h3_0 = node(L[3].head[0])  # recovers 0.57 bits
h3_5 = node(L[3].head[5])  # recovers 0.41 bits
mlp3 = node(L[3].mlp)  # recovers 0.88 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> h3_0,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> h3_0,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    h2_2 >> h3_0,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

