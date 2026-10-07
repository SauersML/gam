"""Behavior greater_than.war (vpd4l; M's top token is right on 96% of the targets): Greater-than: the
end year starts with the same century, so its last two digits must exceed the start year's.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 2.83 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 2.81, L2.H3 1.97, L3.H5 1.74, layer 3 MLP 1.73, L2.H4 1.54, layer 2 MLP
0.51, layer 1 MLP 0.43 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 2.81 bits
mlp1 = node(L[1].mlp)  # recovers 0.43 bits
h2_3 = node(L[2].head[3])  # recovers 1.97 bits
h2_4 = node(L[2].head[4])  # recovers 1.54 bits
mlp2 = node(L[2].mlp)  # recovers 0.51 bits
h3_5 = node(L[3].head[5])  # recovers 1.74 bits
mlp3 = node(L[3].mlp)  # recovers 1.73 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
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
    mlp1 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

