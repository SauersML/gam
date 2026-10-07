"""Behavior greater_than.reign (vpd4l; M's top token is right on 92% of the targets): Greater-than: the
end year starts with the same century, so its last two digits must exceed the start year's.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 2.28 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 2.26, layer 3 MLP 1.60, L2.H3 1.54, L3.H5 1.02, layer 2 MLP 0.85, layer 1
MLP 0.79, L2.H4 0.75, L2.H2 0.27, L2.H1 0.27 bits. The program lets every write among them reach
every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 2.26 bits
mlp1 = node(L[1].mlp)  # recovers 0.79 bits
h2_1 = node(L[2].head[1])  # recovers 0.27 bits
h2_2 = node(L[2].head[2])  # recovers 0.27 bits
h2_3 = node(L[2].head[3])  # recovers 1.54 bits
h2_4 = node(L[2].head[4])  # recovers 0.75 bits
mlp2 = node(L[2].mlp)  # recovers 0.85 bits
h3_5 = node(L[3].head[5])  # recovers 1.02 bits
mlp3 = node(L[3].mlp)  # recovers 1.60 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
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
    mlp1 >> logits,
    h2_1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

