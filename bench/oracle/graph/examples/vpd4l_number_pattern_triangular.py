"""Behavior number_pattern.triangular (vpd4l; M's top token is right on 1% of the targets): Number
patterns: a run of consecutive triangular numbers; the next number continues it.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.01 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 4.99, layer 3 MLP 3.39, L2.H3 3.14, L3.H5 2.76, L3.H0 2.15, L2.H4 1.56,
layer 2 MLP 0.64, L2.H2 0.63, layer 1 MLP 0.54 bits. The program lets every write among them reach
every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 4.99 bits
mlp1 = node(L[1].mlp)  # recovers 0.54 bits
h2_2 = node(L[2].head[2])  # recovers 0.63 bits
h2_3 = node(L[2].head[3])  # recovers 3.14 bits
h2_4 = node(L[2].head[4])  # recovers 1.56 bits
mlp2 = node(L[2].mlp)  # recovers 0.64 bits
h3_0 = node(L[3].head[0])  # recovers 2.15 bits
h3_5 = node(L[3].head[5])  # recovers 2.76 bits
mlp3 = node(L[3].mlp)  # recovers 3.39 bits

edges(
    embed >> mlp0,
    embed >> mlp1,
    embed >> h2_2,
    embed >> h2_3,
    embed >> h2_4,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> mlp1,
    mlp0 >> h2_2,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    mlp1 >> h2_2,
    mlp1 >> h2_3,
    mlp1 >> h2_4,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_2 >> mlp2,
    h2_2 >> h3_0,
    h2_2 >> h3_5,
    h2_2 >> mlp3,
    h2_3 >> mlp2,
    h2_3 >> h3_0,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> mlp2,
    h2_4 >> h3_0,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    mlp1 >> logits,
    h2_2 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

