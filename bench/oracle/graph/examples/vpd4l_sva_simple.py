"""Behavior sva.simple (vpd4l; M's top token is right on 7% of the targets): Subject-verb agreement
after a determiner and noun; the counterfactual flips the subject's number.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 2.52 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 2.51, layer 2 MLP 2.07, L3.H3 1.25, layer 3 MLP 1.08, layer 1 MLP 0.71,
L3.H0 0.65 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 2.51 bits
mlp1 = node(L[1].mlp)  # recovers 0.71 bits
mlp2 = node(L[2].mlp)  # recovers 2.07 bits
h3_0 = node(L[3].head[0])  # recovers 0.65 bits
h3_3 = node(L[3].head[3])  # recovers 1.25 bits
mlp3 = node(L[3].mlp)  # recovers 1.08 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    mlp0 >> mlp3,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    mlp1 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_3 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
    mlp3 >> logits,
)

