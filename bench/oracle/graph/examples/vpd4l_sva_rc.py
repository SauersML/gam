"""Behavior sva.rc (vpd4l; M's top token is right on 3% of the targets): Subject-verb agreement after a
noun followed by a relative clause with a distractor noun; the counterfactual flips the subject's
number.

Nodes: the heads and MLPs whose clean output, patched alone into the counterfactual run, recovers at
least 10% of the 0.80 bits per target token between the clean and the counterfactual next-token
distributions: layer 0 MLP 0.79, layer 2 MLP 0.66, L3.H3 0.62, layer 1 MLP 0.39, L3.H0 0.17, L1.H1
0.16, L2.H0 0.12, L3.H4 0.11 bits. The program lets every write among them reach every later read.
"""
from mech import node, edges, L, embed, logits

mlp0 = node(L[0].mlp)  # recovers 0.79 bits
h1_1 = node(L[1].head[1])  # recovers 0.16 bits
mlp1 = node(L[1].mlp)  # recovers 0.39 bits
h2_0 = node(L[2].head[0])  # recovers 0.12 bits
mlp2 = node(L[2].mlp)  # recovers 0.66 bits
h3_0 = node(L[3].head[0])  # recovers 0.17 bits
h3_3 = node(L[3].head[3])  # recovers 0.62 bits
h3_4 = node(L[3].head[4])  # recovers 0.11 bits

edges(
    embed >> mlp0,
    embed >> h1_1,
    embed >> mlp1,
    embed >> h2_0,
    embed >> mlp2,
    embed >> h3_0,
    embed >> h3_3,
    embed >> h3_4,
    mlp0 >> h1_1,
    mlp0 >> mlp1,
    mlp0 >> h2_0,
    mlp0 >> mlp2,
    mlp0 >> h3_0,
    mlp0 >> h3_3,
    mlp0 >> h3_4,
    h1_1 >> mlp1,
    h1_1 >> h2_0,
    h1_1 >> mlp2,
    h1_1 >> h3_0,
    h1_1 >> h3_3,
    h1_1 >> h3_4,
    mlp1 >> h2_0,
    mlp1 >> mlp2,
    mlp1 >> h3_0,
    mlp1 >> h3_3,
    mlp1 >> h3_4,
    h2_0 >> mlp2,
    h2_0 >> h3_0,
    h2_0 >> h3_3,
    h2_0 >> h3_4,
    mlp2 >> h3_0,
    mlp2 >> h3_3,
    mlp2 >> h3_4,
    mlp0 >> logits,
    h1_1 >> logits,
    mlp1 >> logits,
    h2_0 >> logits,
    mlp2 >> logits,
    h3_0 >> logits,
    h3_3 >> logits,
    h3_4 >> logits,
)

