"""Behavior induction_random.words16 (vpd4l; M's top token is right on 91% of the targets): Induction:
a list of 16 random words is repeated; at a point in the repeat the next word is the one that
followed the same word in the first copy.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 5.75 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 5.73, L2.H4 5.24, layer 3 MLP 3.52, L3.H5 2.50, L2.H3 1.79, L3.H4 1.13,
L3.H0 0.60 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 5.73 bits
h2_3 = node(L[2].head[3])  # recovers 1.79 bits
h2_4 = node(L[2].head[4])  # recovers 5.24 bits
h3_0 = node(L[3].head[0])  # recovers 0.60 bits
h3_4 = node(L[3].head[4])  # recovers 1.13 bits
h3_5 = node(L[3].head[5])  # recovers 2.50 bits
mlp3 = node(L[3].mlp)  # recovers 3.52 bits

edges(
    embed >> mlp0,
    embed >> h2_3,
    embed >> h2_4,
    embed >> h3_0,
    embed >> h3_4,
    embed >> h3_5,
    embed >> mlp3,
    mlp0 >> h2_3,
    mlp0 >> h2_4,
    mlp0 >> h3_0,
    mlp0 >> h3_4,
    mlp0 >> h3_5,
    mlp0 >> mlp3,
    h2_3 >> h3_0,
    h2_3 >> h3_4,
    h2_3 >> h3_5,
    h2_3 >> mlp3,
    h2_4 >> h3_0,
    h2_4 >> h3_4,
    h2_4 >> h3_5,
    h2_4 >> mlp3,
    h3_0 >> mlp3,
    h3_4 >> mlp3,
    h3_5 >> mlp3,
    mlp0 >> logits,
    h2_3 >> logits,
    h2_4 >> logits,
    h3_0 >> logits,
    h3_4 >> logits,
    h3_5 >> logits,
    mlp3 >> logits,
)

