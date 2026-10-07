"""Behavior acronym.three_word (vpd4l; M's top token is right on 58% of the targets): Acronym: after a
three-word capitalized name and an open parenthesis, the next tokens spell its initials.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 4.54 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 4.51, layer 3 MLP 3.28, L2.H3 3.16, L3.H5 2.19, layer 2 MLP 1.32, layer 1
MLP 0.93 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 4.51 bits
mlp1 = node(L[1].mlp)  # recovers 0.93 bits
h2_3 = node(L[2].head[3])  # recovers 3.16 bits
mlp2 = node(L[2].mlp)  # recovers 1.32 bits
h3_5 = node(L[3].head[5])  # recovers 2.19 bits
mlp3 = node(L[3].mlp)  # recovers 3.28 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_3,
    embed >> mlp2,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_3,
    mlp0 >> mlp2,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_3,
    mlp1 >> mlp2,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_3 >> logits,
    mlp2 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

