"""Behavior syllogism.all_are (vpd4l; M's top token is right on 10% of the targets): Syllogism: 'All X
are Y. N is one of the X. So N is a' is followed by Y; the counterfactual changes the category.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 1.31 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 1.30, L2.H3 0.88, L3.H4 0.83, layer 3 MLP 0.66, L2.H2 0.23, L3.H0 0.20,
L3.H5 0.18, layer 2 MLP 0.14 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 1.30 bits
h2_2 = node(L[2].head[2])  # recovers 0.23 bits
h2_3 = node(L[2].head[3])  # recovers 0.88 bits
mlp2 = node(L[2].mlp)  # recovers 0.14 bits
h3_0 = node(L[3].head[0])  # recovers 0.20 bits
h3_4 = node(L[3].head[4])  # recovers 0.83 bits
h3_5 = node(L[3].head[5])  # recovers 0.18 bits
mlp3 = node(L[3].mlp)  # recovers 0.66 bits

edges(
    embed >> mlp0,
    embed >> h2_2,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_4,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

