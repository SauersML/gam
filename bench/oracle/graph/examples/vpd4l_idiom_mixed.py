"""Behavior idiom.mixed (vpd4l; M's top token is right on 83% of the targets): Fixed expression: the
last word of a frequent multiword expression. Phrasings: 'She told me that, {X}' | 'It was, {X}'.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 13.74 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 13.67, layer 2 MLP 8.15, layer 1 MLP 6.85, layer 3 MLP 4.46, L2.H1 2.52,
L3.H5 2.39, L3.H0 1.57, L3.H1 1.56, L1.H1 1.56, L3.H4 1.41 bits. The program lets every write among
them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 13.67 bits
h1_1 = node(L[1].head[1])  # recovers 1.56 bits
mlp1 = node(L[1].mlp)  # recovers 6.85 bits
h2_1 = node(L[2].head[1])  # recovers 2.52 bits
mlp2 = node(L[2].mlp)  # recovers 8.15 bits
h3_0 = node(L[3].head[0])  # recovers 1.57 bits
h3_1 = node(L[3].head[1])  # recovers 1.56 bits
h3_4 = node(L[3].head[4])  # recovers 1.41 bits
h3_5 = node(L[3].head[5])  # recovers 2.39 bits
mlp3 = node(L[3].mlp)  # recovers 4.46 bits

edges(
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_1,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_1,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_1,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_1,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h1_1 >> mlp1,
    h1_1 >> h2_1,
    h1_1 >> mlp2,
    h1_1 >> h3_0,
    h1_1 >> h3_1,
    h1_1 >> h3_4,
    h1_1 >> h3_5,
    h1_1 >> mlp3,
    mlp1 >> h2_1,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_1,
    mlp1 >> h3_4,
    mlp1 >> h3_5,
    mlp1 >> mlp3,
    h2_1 >> mlp2,
    h2_1 >> h3_0,
    h2_1 >> h3_1,
    h2_1 >> h3_4,
    h2_1 >> h3_5,
    h2_1 >> mlp3,
    mlp2 >> h3_0,
    mlp2 >> h3_1,
    mlp2 >> h3_4,
    mlp2 >> h3_5,
    mlp2 >> mlp3,
    h3_0 >> mlp3,
    h3_1 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_1 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_1 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

